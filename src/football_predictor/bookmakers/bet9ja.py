import hashlib

class Bet9jaAdapter:
    @staticmethod
    def generate_booking_code(match_id, prediction, odds, sport="football"):
        # Mock code – replace with real API call later
        code_str = f"B9_{match_id}_{prediction}_{odds}_{sport}"
        code_hash = hashlib.md5(code_str.encode()).hexdigest()[:8].upper()
        return f"B9{code_hash}"
