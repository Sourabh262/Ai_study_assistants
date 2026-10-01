import unittest

from app.exceptions import CalculatorError
from app.tools.calculator import calculate, execute_calculate, format_result


class TestCalculator(unittest.TestCase):
    """Unit tests for the safe AST-based calculator."""

    def test_basic_arithmetic(self):
        self.assertEqual(calculate("2 + 3"), 5.0)
        self.assertEqual(calculate("10 - 4"), 6.0)
        self.assertEqual(calculate("3 * 7"), 21.0)
        self.assertEqual(calculate("20 / 4"), 5.0)
        self.assertEqual(calculate("10 % 3"), 1.0)
        self.assertEqual(calculate("2 ** 4"), 16.0)

    def test_order_of_operations_and_parentheses(self):
        self.assertEqual(calculate("2 + 3 * 4"), 14.0)
        self.assertEqual(calculate("(2 + 3) * 4"), 20.0)
        self.assertEqual(calculate("((10 - 2) * 3) / 4"), 6.0)

    def test_percentages(self):
        self.assertEqual(calculate("20% of 500"), 100.0)
        self.assertEqual(calculate("15.5% of 200"), 31.0)

    def test_execute_calculate_formatting(self):
        self.assertEqual(execute_calculate("25 * 4"), "100")
        self.assertEqual(execute_calculate("5 / 2"), "2.5")

    def test_division_by_zero(self):
        with self.assertRaises(CalculatorError):
            calculate("10 / 0")

    def test_modulo_by_zero(self):
        with self.assertRaises(CalculatorError):
            calculate("10 % 0")

    def test_exponent_limit(self):
        with self.assertRaises(CalculatorError):
            calculate("2 ** 101")

    def test_empty_expression(self):
        with self.assertRaises(CalculatorError):
            calculate("")
        with self.assertRaises(CalculatorError):
            calculate("   ")

    def test_rejection_of_arbitrary_code_injection(self):
        """Ensure calculator rejects imports, functions, system calls, and eval."""
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "eval('2+2')",
            "exec('x = 1')",
            "open('/etc/passwd')",
            "lambda x: x",
            "[x for x in range(10)]",
            "print('hello')",
        ]
        for expr in dangerous_expressions:
            with self.assertRaises(CalculatorError, msg=f"Failed to reject: {expr}"):
                calculate(expr)


if __name__ == "__main__":
    unittest.main()
