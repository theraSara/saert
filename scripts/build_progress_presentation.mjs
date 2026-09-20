// Uses the bundled artifact-tool runtime. See presentations/README.md for reproduction.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root = process.cwd();
const modules = process.env.RUNTIME_NODE_MODULES;
const skill = process.env.SKILL_DIR;
if (!modules || !skill) throw Error('Set RUNTIME_NODE_MODULES and SKILL_DIR');
const {Presentation, PresentationFile} = await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const build = path.join(root,'.presentation_build');
const output = path.join(root,'presentations','2026-09-21-progress');
const data = JSON.parse(await fs.readFile(path.join(build,'data.json'),'utf8'));
const P = data.palette;
const family = resolvePresentationFont();
const p = Presentation.create({slideSize:{width:1280,height:720}});
const notes = [], tableOwners = [], chartOwners = [];
const fmt = n => `${n>=0?'+':''}${n.toFixed(2)}`;
function text(s,value,x,y,w,h,size=28,color=P.ink,bold=false) {
  const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  t.text=value; t.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};
  return t;
}
function slide(title,note,source,footer='') {
  const s=p.slides.add(); s.background.fill=P.paper;
  text(s,title,64,44,1152,100,42,P.plum,true);
  if(footer)text(s,footer,64,638,1080,60,18,P.muted);
  text(s,String(p.slides.items.length),1175,657,40,35,18,P.muted);
  s.speakerNotes.textFrame.setText(`${note}\n\nSources: ${source}`);
  notes.push({title,note,source});return s;
}
function table(s,values,widths,{top=180,height=330,font=25,emphasize=[]}={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:64,top,width:1152,height,values,columnWidths:widths});
  t.styleOptions={headerRow:false,bandedRows:false};
  t.borders.assign({style:'solid',fill:P.paper,width:2});
  for(let r=0;r<values.length;r++) {
    t.rows[r].height=height/values.length;
    for(let c=0;c<values[r].length;c++) {
      const cell=t.getCell(r,c);
      cell.fill=r===0?P.plum:(emphasize.includes(c)?'#F1E8EF':(r%2?'#F7F5F6':P.paper));
      cell.text.style={typeface:family,fontSize:font,color:r===0?P.paper:P.ink,bold:r===0||emphasize.includes(c)};
    }
  }
  tableOwners.push(p.slides.items.length);return t;
}
function statement(s,label,body,y,color=P.green) {
  text(s,label,64,y,1152,44,29,color,true);
  text(s,body,64,y+48,1120,78,27);
}
const regression='results/sae_regression_v2/diagnostics/primary_results.csv; selected_hooks.csv; docs/sae_regression_methods.md';
const sensitivity='configs/scaling_sensitivity.json; results/scaling_sensitivity_v1/comparison.csv; src/scaling_sensitivity.py';

// 1
{
const s=p.slides.add();s.background.fill=P.paper;
text(s,'SparseRT',64,158,1152,100,76,P.plum,true);
text(s,'Sparse language-model features\nand human reading times',64,276,1120,140,44,P.ink);
text(s,'Research progress meeting\n21 September 2026',64,522,1100,76,26,P.green);
const note='Opening, about 30 seconds: I have built a corrected pipeline from reading-time data to sparse GPT-2 features and tested whether they improve prediction on unseen stories. We now have predictive evidence, a meaningful difference between corpora, and a new scaling check. The next question is what drives the gains before we interpret features or intervene on them.';
s.speakerNotes.textFrame.setText(note+'\nSources: docs/project_plan.md; docs/sae_prediction_findings.md');
notes.push({title:'SparseRT',note,source:'docs/project_plan.md; docs/sae_prediction_findings.md'});
}
// 2
{
const s=slide('Research question and evidence levels','About 50 seconds: The professor’s question concerns features that may be weakly used or removed. I separate three claims. Prediction asks whether features help explain held-out RTs. Interpretation asks what those features track. Intervention asks what changes in the model when we edit them. We have completed an initial predictive comparison. We have not established a causal mechanism in human readers.','docs/project_plan.md','Editing GPT-2 changes model predictions. It does not change the observed human RTs.');
statement(s,'Prediction: initial results complete','Do SAE features add information beyond lexical controls and surprisal?',166);
statement(s,'Interpretation: next research phase','Which reproducible features track independently validated linguistic patterns?',310);
statement(s,'Intervention: planned','Do controlled feature changes alter model predictions and RT alignment?',454);
}
// 3
{
const s=slide('Data and outcome definition','About 45 seconds: Provo and Natural Stories provide complementary measurements, but they also differ in texts and participants. We cannot attribute the result difference entirely to reading modality. The current outcome is the log of the arithmetic mean RT for each displayed region or word, not the average of individual log RTs. Participant-level modeling is a later robustness analysis.','docs/baseline_methods.md; docs/project_plan.md','Current analysis predicts word/region means. Participant-level uncertainty remains to be assessed.');
table(s,[['Corpus','Units','Texts','Reading-time measure'],['Provo','2,743 regions','55','First fixation duration'],['Natural Stories','10,256 words','10','Self-paced reading time']],[250,220,140,542],{height:240,font:28});
text(s,'Outcome: ln(arithmetic mean RT in milliseconds)',64,468,1152,58,30,P.green,true);
text(s,'Full stimulus order preserves model context and word alignment.',64,542,1152,65,27);
}
//4
{
const s=slide('Workflow and current progress','About 60 seconds: This is the project map. The first four stages are implemented. I keep the language-model probability path separate from the representation path. The final two stages need additional evidence, rather than automatically labeling a predictive feature as a cognitive process. Every stage saves reproducible artifacts and metadata.','docs/project_plan.md; run.sh','Completed outputs include aligned rows, pinned model/SAE revisions, fold metadata and saved predictions.');
table(s,[['Stage','Main output','Status'],['1  Prepare corpora','Aligned words, RTs and controls','Complete'],['2  Extract GPT-2 states','Surprisal, before/after-word states','Complete'],['3  Apply SAEs','Sparse features and fidelity checks','Complete / sampled fidelity'],['4  Evaluate prediction','Nested story-held-out comparisons','Complete + scaling follow-up'],['5  Interpret features','Stable candidates and linguistic tests','Next'],['6  Intervene in GPT-2','Controlled changes in RT alignment','Planned']],[285,545,322],{top:152,height:448,font:23});
}
//5
{
const s=slide('A fair comparison beyond surprisal','About 60 seconds: Each model predicts the same original outcome on the same held-out words. The high-dimensional models retain unpenalized controls and surprisal, with ridge only on the added representation. We hold out one whole text at a time and tune the penalty inside three grouped training folds. We also select the reported hook inside training data. We do not pick the best test-layer result.','docs/sae_regression_methods.md; configs/sae_regression.json','Outer folds: 55 Provo / 10 Natural Stories. Inner validation: 3 grouped folds.');
table(s,[['Model','Predictors','Question'],['M0','Lexical and position controls','Reference'],['M1','M0 + surprisal and two lags','Does surprisal help?'],['M2','M1 + dense hidden state','Do representations help?'],['M3','M1 + SAE features','Do sparse features help?']],[120,580,452],{top:158,height:355,font:25});
text(s,'Before-word and after-word states form separate analyses.',64,551,1152,58,29,P.green,true);
}
//6
{
const rows=data.baseline.filter(r=>r.Model.includes('spillover') && r.Model.startsWith('128'));
if(rows.length!==2)throw Error('Expected two matched-context baseline rows');
const s=slide('Surprisal adds useful predictive information','About 45 seconds: The corrected baseline contradicts the original weak-surprisal narrative. With lexical and position controls and two spillover lags, surprisal adds held-out predictive value in both datasets. These gains are over controls only. Later SAE gains use the stronger controls-plus-surprisal baseline. The model has a 1,024-token limit, but the primary representation comparison uses matched 128-token windows.','results/baseline_bos/diagnostics/model_comparison.csv; docs/baseline_methods.md','95% intervals resample texts with predictions fixed. They exclude refitting and participant uncertainty.');
table(s,[['Corpus','Surprisal gain (R² pp)','95% interval (pp)'],...rows.map(r=>[r.Corpus.startsWith('Provo')?'Provo FFD':'Natural Stories SPR',fmt(r['Delta R2 (pp)']),r['95% interval (pp)']])],[360,385,407],{height:230,font:29,emphasize:[1]});
text(s,'Matched-context baseline: 128-token windows with 64-token stride',64,475,1152,80,29,P.green,true);
text(s,'The separate 1,024-token context analysis does not establish an advantage.',64,559,1152,62,26);
}
//7
{
const s=slide('SAE gains differ across corpora','About 75 seconds: All bars show gains beyond controls and surprisal. SAE features add predictive value in both corpora, with larger after-word gains. Provo favors the SAE readout. Natural Stories has larger dense-state gains. After-word features observe the target word, so their gain can include lexical information and is not evidence of anticipation. The next slide gives the paired comparison and its uncertainty.','results/sae_regression_v2/diagnostics/primary_results.csv','R² percentage points above M1. Hook and penalty selected within training stories.');
const categories=['Provo before','Provo after','Natural Stories before','Natural Stories after'];
const ch=s.charts.add('bar',{position:{left:64,top:154,width:1152,height:435},categories,
 series:[{name:'Dense state',values:data.primary.map(r=>Number(r['Dense gain (pp)'].toFixed(2))),fill:P.green,valuesFormatCode:'0.00'},
 {name:'SAE features',values:data.primary.map(r=>Number(r['SAE gain (pp)'].toFixed(2))),fill:P.plum,valuesFormatCode:'0.00'}],
 barOptions:{direction:'column',grouping:'clustered',gapWidth:100,overlap:0},hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:24,typeface:family}},
 xAxis:{textStyle:{fontSize:21,typeface:family},majorGridlines:null},
 yAxis:{min:0,max:8,majorUnit:2,numberFormatCode:'0',textStyle:{fontSize:21,typeface:family},majorGridlines:{fill:P.grid,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,typeface:family,bold:true}},
 chartFill:P.paper,plotAreaFill:P.paper});applyPresentationChartFont(ch,{fontFamily:family});chartOwners.push(p.slides.items.length);
}
//8
{
const s=slide('The paired comparison favors different readouts','About 60 seconds: Positive differences favor SAE. Negative differences favor dense states. Provo favors SAE for both state timings. Natural Stories favors dense states after the word, while its before-word interval includes zero. These conditional intervals do not include model-refitting variation or multiple-comparison correction. This result supports a representation comparison, not a general claim that sparse models are more human-like.','results/sae_regression_v2/diagnostics/primary_results.csv','Paired 95% text-bootstrap intervals conditional on saved predictions. Natural Stories has only 10 stories.');
table(s,[['Corpus / state','SAE minus dense (pp)','95% interval'],...data.primary.map(r=>[`${r.Corpus.split(' · ')[0]} / ${r.State.toLowerCase()}`,fmt(r['SAE minus dense (pp)']),r['Difference 95% interval']])],[470,340,342],{top:176,height:344,font:25,emphasize:[1]});
text(s,'Positive favors SAE. Negative favors the dense state.',64,557,1152,60,29,P.green,true);
}
//9
{
const s=slide('Provo points toward a lexical explanation','About 60 seconds: In every Provo outer fold, inner validation selects L01 for the SAE model, before and after the word. This hook is before the first transformer block. It contains token and position information before contextual mixing by the transformer blocks. That is a strong reason to test lexical and position explanations before naming a syntax or integration mechanism. Natural Stories uses later hooks, but that does not isolate reading modality as the cause.','results/sae_regression_v2/diagnostics/selected_hooks.csv; src/surprisal.py','L01 denotes blocks.0.hook_resid_pre. There are 13 recorded hooks across 12 transformer blocks.');
text(s,'55 / 55',64,178,500,110,76,P.plum,true);
text(s,'Provo folds select the earliest SAE hook\nfor both before-word and after-word models.',64,300,1120,100,31);
text(s,'Next control: token identity and position',64,452,1152,56,32,P.green,true);
text(s,'A predictive gain at this hook does not identify syntactic integration.',64,533,1152,66,28);
}
//10
{
const s=slide('Fidelity checks constrain future interventions','About 55 seconds: Applying an SAE and replacing the hidden state with its reconstruction can alter the model. In a small fixed window sample, preserving each rolling-window start sharply reduces reconstruction damage at early Natural Stories hooks. This is a boundary sensitivity, and not proof that every state reconstructs faithfully. We need broader fidelity tests before attributing a behavioral change to one feature.','results/sae_bos/diagnostics/behavior_summary.csv; docs/sae_methods.md','Added NLL in bits/token, lower is better. Four sampled windows per corpus. Provo policies coincide.');
table(s,[['Natural Stories hook','Preserve actual BOS','Preserve each window start'],['L02','+6.395','+0.032'],['L03','+3.158','+0.093']],[350,350,452],{top:185,height:242,font:28,emphasize:[2]});
text(s,'Final-hook reconstruction still adds loss',64,479,1152,50,30,P.green,true);
text(s,'+0.533 bits/token in Provo and +0.679 in Natural Stories',64,544,1152,65,28);
}
//11
{
const s=slide('One feature exposed a scaling failure','About 60 seconds: In Natural Stories story eight, the earliest-hook feature 16980 has tiny training activations but a much larger held-out activation. Dividing by its tiny training standard deviation creates an extreme input to the linear readout. Saved coefficients reproduce the resulting errors. The original inner selector never chose this hook for Natural Stories, so the main selected-model results are unaffected. This prompted a declared follow-up sensitivity, not deletion of a difficult word.','results/sae_regression_v2/diagnostics/feature_shift_audit.csv','Fixed L01 diagnostic: story 8, saucer after-word and or before-word. Feature 16980.');
table(s,[['Quantity','Observed value'],['Largest training activation','0.000034'],['Training standard deviation','0.00000122'],['Held-out activation','0.777']],[710,442],{top:163,height:284,font:28,emphasize:[1]});
text(s,'Follow-up rule, computed inside every training fold',64,484,1152,52,29,P.green,true);
text(s,'Scale = max(feature SD, median SD of retained features)',64,547,1152,64,29);
}
//12
{
const s=slide('The scaling follow-up retains useful signal','About 75 seconds: I ran the new check on the fixed earliest hook in both corpora, keeping the same nested splits and ridge grid. Provo’s after-word gain stays at about six and a half points. Natural Stories’ catastrophic errors disappear and this hook now adds modest predictive value. Its maximum absolute after-word error drops from 111.88 to 1.10 log milliseconds. This is a targeted exploratory sensitivity after observing a failure. It does not replace the original all-hook comparison or establish a new best hook.','configs/scaling_sensitivity.json; results/scaling_sensitivity_v1/comparison.csv','Fixed L01 only. Exploratory follow-up after observing a failure. Original all-hook results remain unchanged.');
table(s,[['Corpus / state','Gain with scale floor (pp)','95% interval'],...data.sensitivity.map(r=>[`${r.corpus==='provo'?'Provo':'Natural Stories'} / ${r.state==='prefix'?'before':'after'}`,fmt(r.floor_gain_pp),`[${fmt(r.floor_ci_low_pp)}, ${fmt(r.floor_ci_high_pp)}]`])],[455,380,317],{top:168,height:336,font:26,emphasize:[1]});
text(s,'Provo after-word: +6.58 originally, +6.53 with the floor',64,547,1152,64,29,P.green,true);
}
//13
{
const s=slide('The next research steps','About 70 seconds: The first priority is separating lexical and scale effects from the interpretability claim. Then nominate features using training data, test stability, and validate linguistic labels using independent texts and minimal pairs. Only after those steps should we change gains or ablate features in GPT-2. Broad reconstruction checks, identity interventions and matched random-feature controls are necessary to interpret the model effects. Participant-level RT robustness also remains part of the paper plan.','docs/project_plan.md; docs/sae_prediction_findings.md','Next deliverable: a controlled representation comparison, followed by a small set of auditable feature dossiers.');
statement(s,'1  Resolve lexical and scaling explanations','Add token/position controls and a separately declared all-hook scaling sensitivity.',160);
statement(s,'2  Validate candidate features','Check training-fold stability, independent examples and linguistic minimal pairs.',305);
statement(s,'3  Test interventions in GPT-2','Expand fidelity checks, then compare feature changes with identity and random controls.',450);
}
//14
{
const s=slide('Questions for the research meeting','About 60 seconds: I would like the team to agree on the next experiment and on what constitutes a sufficient mechanistic claim. My recommendation is to prioritize lexical controls and scaling robustness, then identify a small reproducible feature set. We should also agree on whether participant-level analysis is necessary before feature interpretation or can run alongside it. A defensible contribution needs to explain a representation effect rather than only show improved prediction.','docs/project_plan.md','Current conclusion: sparse features add RT information, with corpus-dependent tradeoffs against dense states.');
statement(s,'Immediate experiment','Should lexical identity controls and the all-hook scaling check be the next milestone?',170);
statement(s,'Interpretability standard','What linguistic validation and intervention controls would support the intended claim?',328);
statement(s,'Paper scope','When should participant-level RT analysis enter the main evidence?',486);
}
//15
{
const s=slide('Appendix: model and evaluation details','Use only if asked. GPT-2 small is frozen, with pinned checkpoint revisions. There are twelve pre-block hooks and the final post-block residual. Prefix states precede the first subtoken, and post states use the last subtoken. Word surprisal sums subtoken surprisals. BOS supplies first-word conditioning. The primary comparison uses 128/64 windows for both representation and surprisal. Every preprocessing operation in regression fits on training data.','src/surprisal.py; docs/sae_regression_methods.md; configs/sae_regression.json','All-hook analysis: 104 combinations and 3,380 outer fits. Scale-floor follow-up: 4 combinations and 130 outer fits.');
table(s,[['Component','Specification'],['Language model','Frozen GPT-2 small, 12 blocks, 13 hooks'],['Sparse representation','24,576 SAE features at each hook'],['Windows','Primary 128/64, separate surprisal 1,024/512'],['Readout','Ridge on features, unpenalized M1 predictors'],['Penalty grid','100, 10, 1, 0.1, 0.01'],['Uncertainty','2,000 paired text bootstraps, fixed predictions']],[350,802],{top:150,height:448,font:24});
}
//16
{
const s=slide('Appendix: three different meanings of zero','Use only if asked about the professor’s original question. An SAE activation can be zero because of its encoding rule. A regression coefficient can be zero because of the fitting procedure, especially Lasso, but the current ridge readout is not feature selection. Experimental ablation is a separate intervention we choose. These should never be conflated. Multiplying a zero activation by a gain still produces zero, so investigating missing features would require a separately designed injection or patching experiment.','docs/project_plan.md','None of these alone establishes that GPT-2 omits a computation used by human readers.');
statement(s,'Zero SAE activation','The encoder assigns no activation to this feature in this context.',170);
statement(s,'Zero regression coefficient','A prediction model gives the feature no fitted weight. Ridge is not sparse selection.',322);
statement(s,'Feature ablation','An experimenter removes a feature contribution and measures downstream changes.',474);
}
//17
{
const s=slide('Appendix: reproducible evidence','Use if asked where to inspect results. The project saves raw predictions, fold choices and input hashes. The numbered notebooks are the current analysis workflow. Older outputs are superseded. The LaTeX notebook reads validated report tables. The scaling follow-up uses its own outputs and never overwrites the original analysis. Main caveats are aggregate RT, ten Natural Stories groups, exploration of these data, readout-capacity differences and incomplete mechanistic validation.','run.sh; docs/presentation_workflow.md; docs/project_plan.md','Study positioning: computational psycholinguistics with model interpretability. Novelty requires a focused literature comparison.');
table(s,[['Evidence','Location'],['Baseline and fidelity','Notebooks 01 and 02'],['SAE prediction','Notebook 03 and saved outer predictions'],['New scaling follow-up','Notebook 04 and scaling_sensitivity_v1'],['Paper tables','Notebook 90'],['Analysis specifications','configs/ and docs/']],[430,722],{top:178,height:354,font:27});
}

await fs.mkdir(output,{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(output,process.env.PRESENTATION_NAME||'saert-progress.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[family]},
 verifyArtifactToolImport:true,receiptPath:path.join(build,path.basename(finalPath)+'.validation.json')});
await fs.writeFile(path.join(output,'speaker-notes.md'), '# SparseRT meeting notes\n\nSuggested pace: slides 1–14 in 12–15 minutes. Slides 15–17 are optional backup.\n\n'+notes.map((n,i)=>`## ${i+1}. ${n.title}\n\n${n.note}\n\nSource: ${n.source}\n`).join('\n'));
console.log(JSON.stringify({finalPath,font:family,slides:p.slides.items.length}));
