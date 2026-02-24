from typing import Dict, List, Optional

class DraftingService:
    """
    Service to manage Lebanese legal document templates and their required fields.
    """
    
    TEMPLATES = {
        "lease_agreement": {
            "name": "عقد إيجار (Lease Agreement)",
            "required_fields": [
                "اسم المؤجر (Lessor Name)",
                "اسم المستأجر (Lessee Name)",
                "وصف المأجور (Property Description)",
                "بدل الإيجار (Rent Amount)",
                "مدة الإيجار (Lease Duration)"
            ],
            "description": "عقد إيجار سكني أو تجاري خاضع لقانون الإيجارات اللبناني."
        },
        "demand_letter": {
            "name": "إنذار بوجوب الدفع (Demand Letter)",
            "required_fields": [
                "اسم الدائن (Creditor Name)",
                "اسم المدين (Debtor Name)",
                "المبلغ المستحق (Amount Owed)",
                "سبب الدين (Reason for Debt)"
            ],
            "description": "كتاب رسمي لمطالبة مدين بتسديد مبالغ متأخرة قبل اتخاذ إجراءات قانونية."
        },
        "power_of_ attorney": {
            "name": "وكالة خاصة (Special Power of Attorney)",
            "required_fields": [
                "اسم الموكل (Principal Name)",
                "اسم الوكيل (Attorney-in-Fact Name)",
                "موضوع الوكالة (Subject of Power of Attorney)"
            ],
            "description": "وكالة لشخص معين للقيام بأعمال قانونية محددة."
        }
    }

    def get_template(self, template_key: str) -> Optional[Dict]:
        return self.TEMPLATES.get(template_key)

    def list_templates(self) -> List[Dict]:
        return [
            {"key": k, "name": v["name"], "description": v["description"]}
            for k, v in self.TEMPLATES.items()
        ]

    def identify_request(self, query: str) -> Optional[str]:
        """
        Identify if the query is a request for a specific document.
        This is a simple keyword-based approach to keep performance high.
        """
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["عقد", "إيجار", "اجار", "lease", "rent"]):
            return "lease_agreement"
        if any(kw in query_lower for kw in ["إنذار", "مطالبة", "demand", "warning"]):
            return "demand_letter"
        if any(kw in query_lower for kw in ["وكالة", "power of attorney", "poa"]):
            return "power_of_attorney"
        return None
