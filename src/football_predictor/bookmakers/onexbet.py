import hashlib

class OnexBetAdapter:
    @staticmethod
    def generate_booking_code(match_id, prediction, odds, sport="football"):
        code_str = f"1X_{match_id}_{prediction}_{odds}_{sport}"
        code_hash = hashlib.md5(code_str.encode()).hexdigest()[:8].upper()
        return f"1X{code_hash}"
