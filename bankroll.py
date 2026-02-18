"""Simple bankroll/stake calculator for single-event betting.

Run:
  python bankroll.py
"""

from __future__ import annotations


def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1.0 else 0.5


def kelly_fraction(prob: float, odds: float) -> float:
    """Kelly fraction for decimal odds."""
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f = (prob * b - q) / b
    return max(0.0, f)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ask_float(prompt: str, default: float | None = None, min_value: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Введите число.")
                continue

        if min_value is not None and val < min_value:
            print(f"Значение должно быть >= {min_value}")
            continue
        return val


def main() -> None:
    print("=== Bankroll calculator ===")
    print("Введите параметры для расчета ставки на выбранный исход.\n")

    bankroll = ask_float("Банкролл: ", min_value=0.01)
    odds = ask_float("Коэффициент (decimal, например 1.85): ", min_value=1.01)
    prob = ask_float("Ваша вероятность исхода (0..1, например 0.57): ", min_value=0.0)

    if prob > 1.0:
        print("Вероятность должна быть в диапазоне 0..1")
        return

    kelly_fraction_used = ask_float("Доля Келли (по умолчанию 0.5): ", default=0.5, min_value=0.0)
    max_stake_pct = ask_float("Макс % от банкролла на ставку (по умолчанию 0.03): ", default=0.03, min_value=0.0)
    min_edge = ask_float("Минимальный edge для ставки (по умолчанию 0.015): ", default=0.015)

    market_prob = implied_prob(odds)
    edge = prob - market_prob

    kelly_full = kelly_fraction(prob, odds)
    kelly_used = kelly_full * kelly_fraction_used
    stake_fraction = clamp(kelly_used, 0.0, max_stake_pct)
    stake_amount = bankroll * stake_fraction

    decision = "BET" if edge >= min_edge and stake_amount > 0 else "NO_BET"

    print("\n=== Result ===")
    print(f"Market implied probability: {market_prob:.4f}")
    print(f"Your edge: {edge:+.4f}")
    print(f"Kelly full fraction: {kelly_full:.4f}")
    print(f"Kelly used fraction: {kelly_used:.4f}")
    print(f"Final stake fraction (after cap): {stake_fraction:.4f}")
    print(f"Recommended stake: {stake_amount:.2f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
