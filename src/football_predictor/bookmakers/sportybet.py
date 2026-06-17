from __future__ import annotations

import hashlib


class SportyBetAdapter:
    @staticmethod
    def generate_booking_code(
        match_id: str | int,
        prediction: str,
        odds: float | str,
        sport: str = "football",
    ) -> str:
        """Generate a mock SportyBet booking code.

        In production this would call SportyBet's bet-slip API (if publicly
        available) and return the real code. Until then a deterministic hash
        over the key fields gives a stable, decodable-in-principle code that
        is safe to demo with real match IDs.
        """
        code_str = f"{match_id}{prediction}{odds}{sport}"
        code_hash = hashlib.md5(code_str.encode()).hexdigest()[:10].upper()
        return f"SB{code_hash}"

    @staticmethod
    def decode_booking_code(code: str) -> None:
        """Reverse lookup placeholder — MD5 is one-way; a real implementation
        would query a stored mapping or SportyBet's API."""
        pass
