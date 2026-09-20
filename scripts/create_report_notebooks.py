"""Build the portable report notebooks. Does not fit models. Overwrites notebook sources."""
from pathlib import Path
import nbformat as n
ROOT=Path(__file__).resolve().parents[1]
md,code=n.v4.new_markdown_cell,n.v4.new_code_cell
SETUP='''from pathlib import Path
import os, sys
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.report_bundle import read_summary, verify_bundle
from src.utils import focus_table, display_details
REPORTS = Path(os.environ.get("SAERT_REPORTS_DIR", ROOT/"reports"))
verify_bundle(REPORTS)
def load(section, name): return read_summary(section, name, REPORTS)
def figure(section, name): display(Image(filename=str(REPORTS/section/(name+".png"))))
'''
def write(name,cells):
    nb=n.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'display_name':'Python (SparseRT reports)','language':'python','name':'python3'}
    n.write(nb,ROOT/'notebooks'/name)

def main():
    write('01_surprisal_baseline.ipynb',[
      md('# Does surprisal improve held-out reading-time prediction?\n\nThis portable notebook reads the verified summaries in `reports/`. It requires no raw corpus, GPU, model download or PC-specific path. Recompute the underlying analysis using `run.sh`; see the README. The main quantity is **R² gain in percentage points beyond controls**, alongside its uncertainty.'),code(SETUP),
      code('''full = load("baseline", "model_comparison")
main = full[full.Model.str.contains("spillover")].copy()
provo = main.Corpus.str.startswith("Provo")
p = main[provo].iloc[:1].copy(); p["Model"] = "Both contexts (identical)"
main = pd.concat([p,main[~provo]])
display(focus_table(main[["Corpus","Model","Delta R2 (pp)","95% interval (pp)"]],
 important=["Delta R2 (pp)"],signed_columns=["Delta R2 (pp)"],formats={"Delta R2 (pp)":"{:+.2f}"}))
figure("baseline","heldout_improvement")'''),
      md('Intervals resample whole texts while holding predictions fixed. They exclude model-refitting and participant uncertainty. Provo’s context regimes coincide; Natural Stories’ context contrast does not establish an advantage.'),
      code('''display_details("Paired contrasts",load("baseline","paired_contrasts"))
display_details("Corpus quality",load("baseline","corpus_quality"))
display_details("Collinearity diagnostics",load("baseline","collinearity"))''')])
    write('02_sae_reconstruction.ipynb',[
      md('# Does SAE reconstruction preserve model behavior?\n\nVector reconstruction error and prediction damage answer different questions. These are sampled fidelity checks, not reading-time effects. Provo’s two boundary policies coincide and appear once.'),code(SETUP),
      code('figure("sae_fidelity","sae_fidelity")'),
      md('Rows count completed transformer blocks: 0 = L01, 1 = L02, and 12 = final residual. Left: relative vector error (%). Right: added NLL (bits/token). The nonlinear color scale preserves visibility across magnitudes; printed numbers remain untransformed. Preserving window starts does not validate the excluded states. Expand this four-window sample before interventions.'),
      code('''display_details("Behavioral fidelity",load("sae_fidelity","behavior_summary"))
display_details("Geometric fidelity",load("sae_fidelity","geometry_summary"))''')])
    write('03_sae_reading_time_prediction.ipynb',[
      md('# Do SAE features add information beyond surprisal?\n\nRead **SAE gain (pp)** first, then its interval and the paired **SAE minus dense** contrast. Both hook and regularization strength are chosen using training stories only. These exploratory aggregate-RT results do not establish feature meaning or human mechanisms.'),code(SETUP),
      code('figure("sae_prediction","heldout_representation_gain")'),
      code('''primary = load("sae_prediction","primary_results")
display(focus_table(primary[["Corpus","State","SAE gain (pp)","SAE 95% interval","Texts improved"]],
 important=["SAE gain (pp)"],signed_columns=["SAE gain (pp)"],formats={"SAE gain (pp)":"{:+.2f}"}))
display(focus_table(primary[["Corpus","State","SAE minus dense (pp)","Difference 95% interval"]],
 important=["SAE minus dense (pp)"],signed_columns=["SAE minus dense (pp)"],formats={"SAE minus dense (pp)":"{:+.2f}"}))'''),
      md('After-word states observe the target word. Confidence intervals condition on fixed predictions and exclude participant/refitting uncertainty. Dense and SAE representations differ in width and nonlinearity.'),
      code('figure("sae_prediction","layer_gain_heatmap")'),
      md('The hook profile is exploratory. Colors saturate at ±10 pp; printed values retain extreme failures. Natural Stories L01 has an audited feature-scaling failure. Inner validation never selected it for the main Natural Stories comparison. Notebook 04 checks an alternative scaling rule.'),
      code('''display_details("Hook selection counts",load("sae_prediction","hook_selection"))
display_details("Regularization grid boundary counts",load("sae_prediction","regularization_summary"))
display_details("Extreme-error audit",load("sae_prediction","feature_shift_audit"))
display_details("All hook metrics",load("sae_prediction","model_comparison"))''')])
    write('04_feature_scaling_robustness.ipynb',[
      md('# Does the earliest-hook result survive a scaling check?\n\nThis is an exploratory response to an observed failure. At fixed L01, each feature scale is floored at the median retained-feature SD, computed inside every training split. No test-derived clipping or row deletion occurs. This does not replace the original all-hook comparison.'),code(SETUP),
      code('''raw = load("scaling_sensitivity","comparison")
view = pd.DataFrame({"Corpus":raw.corpus.map({"provo":"Provo","natural_stories":"Natural Stories"}),
 "State":raw.state.map({"prefix":"Before word","post":"After word"}),"Gain with floor (pp)":raw.floor_gain_pp,
 "95% interval":[f"[{a:+.2f}, {b:+.2f}]" for a,b in zip(raw.floor_ci_low_pp,raw.floor_ci_high_pp)]})
display(focus_table(view,important=["Gain with floor (pp)"],signed_columns=["Gain with floor (pp)"],formats={"Gain with floor (pp)":"{:+.2f}"}))
errors=view[["Corpus","State"]].copy()
errors["Original max error"]=raw.original_max_abs_error
errors["Floor max error"]=raw.floor_max_abs_error
display(focus_table(errors,important=["Floor max error"],formats={"Original max error":"{:.2f}","Floor max error":"{:.2f}"},caption="Worst absolute prediction error in log milliseconds, not average error."))'''),
      md('Provo’s after-word gain is nearly unchanged (+6.58 to +6.53 pp). The extreme Natural Stories first-hook errors disappear. This does not prove optimal scaling or establish what an all-hook comparison would find. Next: lexical identity/position controls and a declared all-hook sensitivity.'),
      code('display_details("Complete comparison, including original failures",raw)')])
    write('90_export_latex_tables.ipynb',[
      md('# LaTeX table workshop\n\nOne notebook for paper tables. It reads portable report CSVs, exports selected columns, and preserves unrounded values and relative-path provenance. Add `\\usepackage{booktabs}` to your paper. Bold marks the primary metric, not significance.'),code(SETUP+'''from src.table_export import export_table
OUTPUT = REPORTS/"tables"
'''),
      code('''catalog = {
 "sae_prediction":dict(section="sae_prediction",name="primary_results",columns=["Corpus","State","SAE gain (pp)","SAE 95% interval"],metric="SAE gain (pp)",caption="SAE gains beyond lexical controls and matched-context surprisal."),
 "sae_dense_contrast":dict(section="sae_prediction",name="primary_results",columns=["Corpus","State","SAE minus dense (pp)","Difference 95% interval"],metric="SAE minus dense (pp)",caption="Paired SAE-versus-dense comparison. Positive values favor SAE."),
 "scaling_sensitivity":dict(section="scaling_sensitivity",name="comparison",columns=["corpus","state","floor_gain_pp","floor_ci_low_pp","floor_ci_high_pp"],metric="floor_gain_pp",caption="Exploratory fixed-L01 sensitivity using a training-derived scale floor.")}
TABLES = list(catalog)  # Choose one or more names.
for name in TABLES:
    spec = catalog[name]; frame=load(spec["section"],spec["name"])
    export_table(frame,OUTPUT/(name+".tex"),columns=spec["columns"],caption=spec["caption"],label="tab:"+name.replace("_","-"),
      decimals={c:2 for c in spec["columns"] if pd.api.types.is_numeric_dtype(frame[c])},signed=[spec["metric"]],emphasize=[spec["metric"]],wide=True,
      notes="Gains in R2 percentage points. Intervals resample texts conditional on saved predictions, excluding participant and refitting uncertainty.",
      sources=[REPORTS/spec["section"]/(spec["name"]+".csv")])
    display_details(name,frame[spec["columns"]])
print("Tables saved under reports/tables")'''),
      md('## Any other table\n\nUse the same exporter with your DataFrame. Specify useful columns, precision, units and uncertainty. The example is disabled until you provide a CSV path.'),
      code('''CUSTOM_CSV = None  # Example: REPORTS/"baseline/model_comparison.csv"
if CUSTOM_CSV is not None:
    custom = pd.read_csv(CUSTOM_CSV)
    export_table(custom,OUTPUT/"custom_table.tex",caption="Replace with your caption",label="tab:custom",sources=[CUSTOM_CSV])''')])
    print('Created five portable notebooks')
if __name__=='__main__':main()
