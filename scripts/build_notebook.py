"""Build the analysis notebook, execute it in a fresh kernel, export HTML."""
from pathlib import Path
import os
import sys
import nbformat as nbf
import nbconvert
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]

def main():
    os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.runtime'))
    # Support isolated --target installs as well as standard virtual environments.
    package_share = Path(nbconvert.__file__).resolve().parents[1] / 'share/jupyter'
    if package_share.is_dir():
        os.environ['JUPYTER_PATH'] = str(package_share) + os.pathsep + os.environ.get('JUPYTER_PATH','')
    if '--export-only' in sys.argv:
        completed = nbf.read(ROOT/'notebooks/appointment_attendance.ipynb', as_version=4)
        assert all(c.execution_count is not None for c in completed.cells if c.cell_type=='code')
        html,_=HTMLExporter().from_notebook_node(completed)
        (ROOT/'results/notebook-report.html').write_text(html,encoding='utf-8')
        print('Previously executed notebook exported; no cells rerun.')
        return
    nb=nbf.v4.new_notebook()
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    nb.cells=[
        md('''# When appointments are missed
## Scheduling lead time, age, and the reminder comparison trap


**Purpose:** describe attendance patterns in a historical Brazilian appointment dataset and show how comparisons change when groups have different scheduling profiles. The unit is an appointment, not a unique patient. No individual records or patient identifiers are displayed.

Questions: How do missed-appointment rates differ across scheduling lead times and broad age bands? Does the raw SMS comparison persist within lead-time bands? How sensitive are summaries to repeated appointments and unusual ages?'''),
        md('''## 1. Reproduce the analysis

The original published CSV is kept locally and verified by SHA-256. The reusable module documents all cleaning, aggregation, and chart code. A fixed seed and 600 patient-cluster bootstrap replicates make the uncertainty summaries reproducible. The notebook only prints aggregates; running it never changes the source CSV.'''),
        code('''from pathlib import Path
import sys
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.analysis import run_analysis
overview, tables = run_analysis()
display(Markdown(f"**Analysis cohort:** {overview['analysis_appointments']:,} appointments from {overview['unique_patients']:,} recorded patient IDs. **Window:** {overview['appointment_start']} to {overview['appointment_end']}. **Missed:** {overview['missed']:,} ({overview['missed_rate']:.2%})."))'''),
        md('''## 2. Establish a defensible cohort

Identifiers are read as strings, not continuous quantities. `No-show = Yes` means the appointment was missed. Dates are parsed as UTC; lead time is the difference between normalized calendar dates because appointment times are recorded at midnight. This is not elapsed-hour waiting time, and local timezone boundaries are not independently verified.

Exclude negative/missing age and invalid or negative calendar-day lead time. Retain age zero and high ages; compare an age≤100 sensitivity cohort rather than declaring all unusual ages invalid. Repeated patients remain in the primary cohort because they have distinct appointments. Duplicate appointment IDs would stop the pipeline for review rather than being silently removed. No missing values are imputed.'''),
        code('display(tables["cleaning-audit"])\ndisplay(tables["descriptive-statistics"].round(2))'),
        md('''## 3. Describe scheduling and age

The rate is missed appointments divided by all appointments in each group. Counts accompany comparisons to show denominators. Broad bins are chosen for readability; they can hide within-bin variation. Bootstrap intervals resample complete patient clusters, accommodating repeated appointments within a patient. They assume sampled patient clusters are exchangeable and do not account for clinic-level dependence or an unrepresentative dataset.'''),
        code('''for key, name in [("lead-time", "01-lead-time.png"), ("age-groups", "02-age-groups.png")]:
    display(tables[key].round(4))
    display(Image(filename=str(ROOT / "results/figures" / name)))
display(Image(filename=str(ROOT / "results/figures/04-distributions.png")))'''),
        md('''Same-day appointments have a much lower missed rate than appointments booked ahead. The pattern is descriptive: same-day and future appointments may differ in urgency, availability, or patient circumstances. It does not prove that shortening lead time would cause this reduction. Rates are highest in the 18–34 band and lower in older bands; no individual attendance decision should be based on these aggregates.'''),
        md('''## 4. Compare like with more-like: SMS and lead time

The overall SMS comparison mixes same-day bookings with appointments scheduled well ahead. The within-band comparison below uses positive lead times only, because no same-day appointments have SMS recorded. Every published stratum has at least 100 appointments.

For a descriptive standardized comparison, compute each SMS group's rate in four positive lead-time bands and weight both groups by the same pooled appointment distribution. This adjusts the presentation for these coarse bands only. It is not a causal effect estimate; message timing, delivery, selection, age, access barriers, and other differences remain uncontrolled.'''),
        code('''display(tables["sms-overall"].round(4))
display(tables["sms-by-lead"].round(4))
display(Image(filename=str(ROOT / "results/figures/03-sms-within-lead.png")))
display(Markdown(f"With a common positive-lead-time distribution, the descriptive rates are **{overview['standardized_sms_rate']:.2%} with SMS recorded** and **{overview['standardized_no_sms_rate']:.2%} without**, a difference of **{overview['standardized_difference_pp']:.2f} percentage points**. These rates apply to the positive-lead-time comparison population, not the full cohort."))'''),
        md('''The direction differs from the raw overall SMS comparison. The groups have different scheduling profiles, so a raw difference can be misleading. Even the standardized comparison cannot demonstrate that reminders prevented missed visits. A randomized or otherwise credible causal design, with delivery timing and patient access context, would be needed for that claim.'''),
        md('''## 5. Challenge the result

Repeat the overview using one earliest-booked valid appointment per recorded patient and using only ages at most 100. The one-per-patient cohort changes the population and should be treated as a sensitivity analysis, not a corrected estimate. Earliest-booked ties are broken deterministically by appointment date and ID. Compare the lead-time pattern as well as the overall percentage.'''),
        code('display(tables["sensitivity"].round(4))\ndisplay(tables["first-appointment-lead"].round(4))'),
        md('''## 6. What this supports—and what remains unknown

The data support a retrospective description: lead time is strongly associated with attendance, and group composition changes the SMS story. A useful operational follow-up is to understand scheduling barriers and then evaluate a clearly defined intervention, not to classify or penalize patients using this dataset.

Limits: a short historical 2016 window; unknown sampling/coverage; repeated appointments; uncertain timestamp semantics; coarse bands; unmeasured confounding; unknown reminder timing and delivery; no causal design; and no evidence of generalization to another setting. The ambiguous `Handcap` coding is retained in the source but not analyzed. This is not a clinical prediction or individual decision tool.

**Sources:** [published CSV snapshot](https://video.udacity-data.com/topher/2025/September/68c92f00_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv), [original Kaggle listing](https://www.kaggle.com/datasets/joniarroba/noshowappointments). Redistribution terms were not verifiable in accessible metadata, so raw data are excluded from the repository.

''')
    ]
    nb.metadata={'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
                 'language_info':{'name':'python','version':sys.version.split()[0]}}
    target=ROOT/'notebooks/appointment_attendance.ipynb'
    nbf.write(nb,target)
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.validate(nb)
    for cell in nb.cells:
        cell.metadata.pop("execution",None)
    nbf.write(nb,target)
    exporter=HTMLExporter()
    html,_=exporter.from_notebook_node(nb)
    (ROOT/'results/notebook-report.html').write_text(html,encoding='utf-8')
    print('Fresh notebook execution completed; HTML exported; all output cells checked for errors.')

if __name__=='__main__':
    main()
