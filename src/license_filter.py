import re
import logging

logger = logging.getLogger(__name__)

class LicenseFilter:
    def __init__(self, allowed_cc: list, allow_nd: bool):
        self.allowed_cc = [cc.lower() for cc in allowed_cc]
        self.allow_nd = allow_nd

    def evaluate(self, license_url: str) -> tuple[bool, str, str]:
        """Trả về (is_valid, normalized_license, reason)"""
        if not license_url:
            return False, "", "Empty license URL"

        match = re.search(r'creativecommons\.org/licenses/([^/]+)/', license_url.lower())
        if not match:
            return False, "", "Not a recognized Creative Commons URL"

        cc_type = match.group(1) # e.g., 'by-nc-sa', 'by-nd'

        if cc_type not in self.allowed_cc:
            return False, cc_type, f"License '{cc_type}' is not in allowed list: {self.allowed_cc}"

        if not self.allow_nd and "nd" in cc_type.split("-"):
            return False, cc_type, "Derivative works (ND) are prohibited by config"

        return True, cc_type, ""