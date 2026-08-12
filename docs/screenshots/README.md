# Application screenshot guide

Capture screenshots from a clean deterministic demo run at desktop width (recommended
1440×900 or larger). Do not use mock data outside the application or include local paths,
credentials, patient-identifying data, or unrelated browser UI.

Recommended shots:

1. New Case before execution, showing `DEMO` mode and the safety notice.
2. Neurological investigation on Workflow, showing the rendered graph, completed Neurology,
   and Cardiology/Infectious Disease as not selected.
3. Evidence tab with the source title, excerpt, similarity, and public URL.
4. Human Review with the emergency triage warning and decision controls.
5. Approved Final Report with the mandatory clinician-review disclaimer.

Launch with `streamlit run app.py`, use a bundled synthetic case, and avoid committing local
SQLite databases or logs with the images.
