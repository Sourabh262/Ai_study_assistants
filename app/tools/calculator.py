import ast
import operator
import re

from app.exceptions import CalculatorError
from app.logging_config import logger


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp):
        operation = _ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise CalculatorError("Unsupported mathematical operator.")

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalculatorError("Exponent is too large.")

        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalculatorError("Division by zero is not allowed.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise CalculatorError("Unsupported unary operator.")

        return operation(_evaluate_node(node.operand))

    raise CalculatorError("Invalid mathematical expression.")


def calculate(expression: str) -> float:
    """
    Safely evaluate a basic mathematical expression.

    Supported:
    +, -, *, /, %, **, parentheses
    """

    if not expression or not expression.strip():
        raise CalculatorError("Mathematical expression cannot be empty.")

    expression = expression.strip()

    # Support simple expressions such as:
    # "20% of 500" -> "500 * 20 / 100"
    percent_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
        expression,
        flags=re.IGNORECASE,
    )

    if percent_match:
        percentage = float(percent_match.group(1))
        value = float(percent_match.group(2))
        result = value * percentage / 100

        logger.info(
            "Calculator: %s%% of %s = %s",
            percentage,
            value,
            result,
        )

        return result

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree.body)

        logger.info(
            "Calculator: %s = %s",
            expression,
            result,
        )

        return result

    except CalculatorError:
        raise

    except (SyntaxError, ValueError, TypeError) as exc:
        logger.error(
            "Invalid calculator expression '%s': %s",
            expression,
            exc,
        )
        raise CalculatorError(
            "Invalid mathematical expression."
        ) from exc

    except Exception as exc:
        logger.error(
            "Calculator failed for '%s': %s",
            expression,
            exc,
        )
        raise CalculatorError(
            "Failed to calculate the expression."
        ) from exc


def format_result(result: float) -> str:
    """Return a clean human-readable calculator result."""

    if result.is_integer():
        return str(int(result))

    return f"{result:.10f}".rstrip("0").rstrip(".")