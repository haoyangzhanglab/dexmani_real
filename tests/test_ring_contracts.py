"""共享最新快照的字段、所有权及跨进程一致性契约。"""

import multiprocessing as mp
import platform
import unittest

import numpy as np

from dexmani_real.ipc.ring import SharedRing

DTYPE = np.dtype([('generation', 'u8'), ('left', 'i8', (128,)), ('right', 'i8', (128,))])


def write_generations(ring, start):
    start.wait()
    for generation in range(2, 2001):
        ring.write(generation=generation, left=generation, right=-generation)


def read_generations(ring, start, result):
    try:
        initial, sequence = ring.read()
        assert sequence == 1 and initial["generation"] == 1
        result.send("ready")
        start.wait()
        for _ in range(4000):
            snapshot = ring.read()
            if snapshot is not None:
                data, sequence = snapshot
                assert int(data['generation']) == sequence
                assert np.all(data['left'] == sequence)
                assert np.all(data['right'] == -sequence)
        result.send(None)
    except BaseException as error:  # noqa: BLE001 — 子进程将失败传回测试父进程。
        result.send(repr(error))
    finally:
        result.close()


class RingContracts(unittest.TestCase):
    def setUp(self):
        self.ctx = mp.get_context('spawn')
        self.ring = SharedRing(self.ctx, DTYPE, 2)

    def test_capacity_dtype_and_fields(self):
        for capacity in (0, -1, True, 1.5, '2', None):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, 'capacity'):
                SharedRing(self.ctx, DTYPE, capacity)
        for dtype in ('f4', [('x', object)], [('x', [('y', object)])]):
            with self.assertRaises(ValueError):
                SharedRing(self.ctx, dtype, 2)
        for populated in (False, True):
            if populated:
                self.ring.write(generation=1, left=1, right=-1)
            with self.assertRaisesRegex(ValueError, 'unknown ring fields'):
                self.ring.read(fields=('typo',))
        for values in ({'generation': 2}, {'generation': 2, 'left': 2, 'right': -2, 'extra': 1}):
            with self.assertRaises(ValueError):
                self.ring.write(**values)
        self.assertEqual(self.ring.latest_sequence, 1)

    def test_readonly_cache_and_subset(self):
        self.assertIsNone(self.ring.read())
        self.ring.write(generation=1, left=1, right=-1)
        data, sequence = self.ring.read()
        original = data['left']
        with self.assertRaises(ValueError):
            original[0] = 99
        data['generation'] = 99
        del data['right']
        subset, subset_sequence = self.ring.read(fields=('left',))
        self.assertEqual(set(subset), {'left'})
        self.assertIs(subset['left'], original)
        full, full_sequence = self.ring.read()
        self.assertEqual((sequence, subset_sequence, full_sequence), (1, 1, 1))
        self.assertIs(full['left'], original)
        self.assertEqual(full['generation'], 1)
        self.assertIn('right', full)
        for generation in range(2, 7):
            self.assertEqual(self.ring.write(generation=generation, left=generation, right=-generation), generation)
            latest, seq = self.ring.read()
            self.assertEqual(seq, generation)
            self.assertTrue(np.all(latest['left'] == generation))
        self.assertTrue(np.all(original == 1))

    def test_nested_structured_field_is_readonly(self):
        ring = SharedRing(self.ctx, [('record', [('value', 'i8'), ('array', 'f4', (2,))])], 1)
        ring.write(record=(1, [2, 3]))
        record = ring.read()[0]['record']
        with self.assertRaises(ValueError):
            record['value'] = 9
        with self.assertRaises(ValueError):
            record['array'][0] = 9

    def test_incomplete_publication_returns_only_previous_complete_snapshot(self):
        ring = SharedRing(self.ctx, DTYPE, 1)
        ring.write(generation=1, left=1, right=-1)
        cached = ring.read()
        ring.write(generation=2, left=2, right=-2)
        # 第三次写入覆盖最新槽，但尚未发布；最新序列仍为 2，读缓存为 1。
        slot = ring.slots[0]
        slot['sequence'] = 5
        slot['data']['generation'] = 3
        data, sequence = ring.read()
        self.assertEqual(sequence, cached[1])
        self.assertEqual(data['generation'], 1)
        self.assertTrue(np.all(data['left'] == 1))
        subset, sequence = ring.read(fields=('left',))
        self.assertEqual(sequence, 1)
        self.assertEqual(set(subset), {'left'})
        subset.clear()
        self.assertEqual(set(ring.read()[0]), set(DTYPE.names))
        ring._last_snapshot = None
        self.assertIsNone(ring.read())

    @unittest.skipUnless(platform.system() == 'Linux' and platform.machine() == 'x86_64',
                         'Ring protocol assumes Linux x86_64')
    def test_concurrent_writer_and_readers(self):
        self.ring.write(generation=1, left=1, right=-1)
        start = self.ctx.Event()
        pipes = [self.ctx.Pipe(duplex=False) for _ in range(2)]
        processes = [self.ctx.Process(target=write_generations, args=(self.ring, start))]
        processes += [self.ctx.Process(target=read_generations, args=(self.ring, start, send))
                      for _, send in pipes]
        try:
            for process in processes:
                process.start()
            for receive, _ in pipes:
                self.assertTrue(receive.poll(30), "reader did not become ready")
                self.assertEqual(receive.recv(), "ready")
            start.set()
            for receive, _ in pipes:
                self.assertTrue(receive.poll(30), 'reader did not finish')
                self.assertIsNone(receive.recv())
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            data, sequence = self.ring.read()
            self.assertEqual(sequence, 2000)
            self.assertTrue(np.all(data['left'] == sequence))
            self.assertTrue(np.all(data['right'] == -sequence))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                process.close()
            for receive, send in pipes:
                receive.close()
                send.close()
