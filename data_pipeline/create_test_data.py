import pandas as pd
import json

data = [
    {
        "source_type": "article",
        "text_content": "Article 1: All Lebanese are equal before the law. They enjoy equal civil and political rights and are equally bound by public obligations and duties without any distinction.",
        "metadata": json.dumps({"law_type": "constitutional", "article_num": 1})
    },
    {
        "source_type": "ruling",
        "text_content": "Ruling 2023/45: In cases of commercial dispute involving international parties, the Lebanese courts shall prioritize the terms of the signed arbitration clause over local standard procedures.",
        "metadata": json.dumps({"court_level": "supreme", "case_year": 2023})
    },
    {
        "source_type": "article",
        "text_content": "Article 562: The penal code provides specific mitigating circumstances for crimes committed in a state of high passion, although recent amendments have significantly restricted these applications.",
        "metadata": json.dumps({"law_type": "penal", "article_num": 562})
    }
]

df = pd.DataFrame(data)
df.to_excel("mock_lebanese_laws.xlsx", index=False)
print("Created mock_lebanese_laws.xlsx")
