"""CPU-only collection contract tests; run with assignment1's Python."""
import unittest
import matmul08 as m


class CoverageTests(unittest.TestCase):
    def test_seven_numbered_kernels_at_each_size(self):
        for n in (1024, 2048, 4096):
            runs = m.coverage_runs(n)
            self.assertEqual(set(runs), {f'k{k}' for k in range(8, 15)})
            for variant, config, stages in runs.values():
                d = dict(m.CONFIGS[config], STAGES=stages, **m.VARIANTS[variant])
                self.assertIsNone(m.why_invalid(d, 227 * 1024))
                self.assertEqual(n % d['BM'], 0)
                self.assertEqual(n % d['BN'], 0)

    def test_missing_matrix_report_rejected(self):
        with self.assertRaisesRegex(ValueError, 'matmul_streamk'):
            m.validate_profile_csv('"Kernel Name","gpu__time_duration.sum"\n"","ns"\n"zero_tiles","100"\n', ['matmul_streamk'])

    def test_streamk_setup_and_matrix_are_both_recognized(self):
        names = m.validate_profile_csv('"Kernel Name","gpu__time_duration.sum"\n"","ns"\n"zero_tiles","100"\n"matmul_streamk","200"\n', ['matmul_streamk', 'zero_tiles'])
        self.assertEqual(len(names), 2)

    def test_opaque_cublas_kernel_name_accepted(self):
        names = m.validate_profile_csv('"Kernel Name","gpu__time_duration.sum"\n"","ns"\n"Kernel2","200"\n', [])
        self.assertEqual(names, ['Kernel2'])

    def test_empty_cublas_report_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no launch'):
            m.validate_profile_csv('"Kernel Name","gpu__time_duration.sum"\n"","ns"\n', [])

    def test_unsupported_size_rejected(self):
        with self.assertRaises(ValueError):
            m.coverage_runs(1000)


if __name__ == '__main__':
    unittest.main()
