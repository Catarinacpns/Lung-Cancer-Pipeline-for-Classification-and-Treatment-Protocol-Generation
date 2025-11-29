# Standard library only
import os

def generate_structured_prompt_tnm(t_stage, n_stage, m_stage, histopath_grade, cancer_type, age, gender, additional_info=None):
    """
    Generates a structured medical prompt for the Gemini RAG system to classify lung cancer 
    based on TNM staging and guide an evidence-based treatment strategy.
    Ensures independent logic for NSCLC and SCLC, and guides response structure accordingly.

    Parameters:
    - t_stage (str): Tumor stage (T).
    - n_stage (str): Lymph node involvement (N).
    - m_stage (str): Metastasis stage (M).
    - histopath_grade (str): Tumor differentiation grade.
    - cancer_type (str): Type of lung cancer (e.g., "Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer").
    - age (int): Patient's age.
    - gender (str): Patient's gender.
    - additional_info (str, optional): Other relevant factors (e.g., "Smoker", "Comorbidities").

    Returns:
    - str: A structured and detailed medical prompt for evidence-based reasoning and response generation.
    """

    common_info = f"""
You are a clinical oncology assistant specialized in lung cancer.

Your tasks:
1. **Determine the clinical TNM stage** (I–IV, including substages A, B, or C) based on the AJCC 8th Edition staging system.
2. **Generate a structured, evidence-based treatment plan** according to the stage, histology, and type of lung cancer.
3. **Use only information derived from retrieved clinical guidelines and peer-reviewed literature** to support your reasoning.
4. **Do not assume facts outside the provided information.**
5. **Use specific medical terminology**, and name all **treatments, radiotherapy modalities, and chemotherapy/immunotherapy regimens explicitly** when referenced in guidelines.

---

### Patient Information
- **Type of Cancer:** {cancer_type}
- **Age:** {age}
- **Gender:** {gender}
- **Tumor (T) Stage:** {t_stage}
- **Lymph Node (N) Stage:** {n_stage}
- **Metastasis (M) Stage:** {m_stage}
- **Histopathological Grade:** {histopath_grade}
{f"- **Additional Clinical Factors:** {additional_info}" if additional_info else ""}
"""

    nsclc_prompt = """
---

### TNM Staging Classification (NSCLC)
- Classify the patient’s cancer into the correct clinical stage using the TNM (AJCC 8th Edition) system.
- You must always specify the substage letter (A, B, or C) when reporting the stage. For example: Stage IIA, Stage IIIB, Stage IVA,  Stage IVB.
- Justify the staging using anatomical and clinical criteria from validated guidelines.

---

### Evidence-Based Treatment Strategy (NSCLC)
Structure your response based on the clinical stage:

- **Stage I–II**: Guide curative options such as surgery, stereotactic body radiation therapy (SBRT), neoadjuvant/adjuvant chemotherapy. Evaluate patient fitness and comorbidities.

- **Stage III**: 
    - Distinguish between resectable and unresectable disease. 
    - Guide multimodal approaches including concurrent chemoradiotherapy, neoadjuvant therapy, or surgical resection with adjuvant therapy. 
    - Include biomarker-driven therapies (e.g., EGFR, ALK, ROS1, PD-L1, BRAF).

- **Stage IV**: Guide systemic treatment approaches:
  - Specify line of therapy (first-line, second-line, refractory)
  - Define histologic subtype and performance status
  - Include molecular marker-based treatment (e.g., EGFR inhibitors, ALK inhibitors, PD-L1 checkpoint inhibitors)

- **Non-Surgical Management**: Describe alternative definitive approaches such as SBRT, hypofractionated EBRT, or systemic therapy alone.

- **Clinical Trials**: Indicate when enrollment is recommended.

- **Palliative Care**: Include symptom control, psychosocial support, and advanced care planning.

- **Follow-Up and Surveillance**: Provide guideline-driven recommendations for imaging, biomarker monitoring, and toxicity management.
"""

    sclc_prompt = f"""
---

### TNM and Traditional Stage Classification (SCLC)
- Classify the patient’s cancer using the AJCC 8th Edition TNM system.
- Based on TNM and anatomical considerations, determine whether the patient has:
  - **Limited-Stage SCLC (LS-SCLC)**
  - **Extensive-Stage SCLC (ES-SCLC)**
- Justify classification using validated criteria from clinical staging references.

---

### Evidence-Based Treatment Strategy (SCLC)
Structure your response based on SCLC stage:

- **Limited-Stage SCLC (LS-SCLC)**:
  - Recommend **concurrent chemoradiation** using **etoposide + cisplatin or carboplatin**, combined with **thoracic radiation therapy (TRT)**.
  - Indicate when **surgical resection** (e.g., lobectomy) may be considered for T1–T2, N0 cases.
  - Include use of **prophylactic cranial irradiation (PCI)** in patients with complete/near-complete response to initial treatment.
  - Mention **durvalumab** as consolidation therapy if supported by recent clinical evidence.

- **Extensive-Stage SCLC (ES-SCLC)**:
  - Recommend systemic therapy.
  - Specify use of **thoracic radiation** in responders, and **PCI or MRI surveillance**.

- **Non-Surgical Management**:
  - Emphasize chemotherapy and radiation-based treatment regimens.
  - Provide alternatives (e.g., sequential chemoradiation or chemotherapy alone) for patients with poor performance status or comorbidities.

- **Older Adults (≥70 years)** only if age above 70:
  - Evaluate treatment tolerance based on comorbidities and performance status.
  - Adjust treatment intensity accordingly.
  - Highlight higher risks of **hematologic toxicity** and **treatment-related mortality** with standard chemoradiotherapy.
  - Mention use of **supportive care**, dose-reduction strategies, or monotherapy if clinically indicated.
  - Include data on survival equivalence in older adults who complete standard therapy and caution when extrapolating trial data.

- **Clinical Trials**:
  - Highlight ongoing investigations into immunotherapy, radiotherapy fractionation, and novel agents.

- **Palliative Care**:
  - Include symptom management for brain metastases, superior vena cava syndrome, and paraneoplastic syndromes.
  - Discuss early integration of palliative services.

- **Follow-Up and Surveillance**:
  - Recommend the followup strategies for SCLC based on the stage (mention the time frequence and need for CT scans).
  - Say what needs to be monitored (e.g.,Chemotherapy-related toxicities)
  
"""

    final_questions = """
---

### Final Structured Output
Include the following:

1. **Clinical Stage**: AJCC TNM stage and, if SCLC, Limited or Extensive stage classification.
2. **Treatment Plan**: Structured by stage and supported by current clinical guidelines.
3. **Therapeutic Modalities**: Use specific names of chemotherapy agents, radiotherapy modalities (e.g., PCI, TRT), and immunotherapies.
4. **Clinical Trial Considerations**: Identify trial opportunities based on disease state and patient factors.
5. **Palliative and Supportive Care**: Describe symptom-focused care and when it should be integrated.
6. **Follow-Up Plan**: Provide evidence-based recommendations for surveillance and survivorship care.
"""

    if "small cell carcinoma" in cancer_type.lower():
        prompt = common_info + sclc_prompt + final_questions
    else:
        prompt = common_info + nsclc_prompt + final_questions

    return prompt

