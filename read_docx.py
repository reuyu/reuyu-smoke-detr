from docx import Document
doc = Document('smoke-dert.docx')

output = []
# 모든 paragraph 추출
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if text:
        # RCM, Rectangular, calibration 관련 키워드 찾기
        keywords = ['RCM', 'Rectangular', 'calibration', 'strip', 'pool', 'Avg', 'vertical', 'horizontal', 'Eq (', 'Equation']
        for kw in keywords:
            if kw.lower() in text.lower():
                output.append(f"=== Paragraph {i} (found: {kw}) ===")
                output.append(text)
                output.append("")
                break

with open('rcm_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
    
print(f"Saved {len(output)} lines to rcm_analysis.txt")
