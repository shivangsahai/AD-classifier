const qs = s => document.querySelector(s);
let latestResults = [];
const patient = () => Object.fromEntries(new FormData(qs("#patientForm")));

document.querySelectorAll(".tabs button").forEach(btn => btn.onclick = () => {
  document.querySelectorAll(".tabs button,.tab-body").forEach(x => x.classList.remove("active"));
  btn.classList.add("active"); qs(`#${btn.dataset.tab}Form`).classList.add("active"); qs("#results").innerHTML = "";
});

qs("#manualForm").onsubmit = async e => {
  e.preventDefault(); setStatus("Analysing EEG values...");
  const data = Object.fromEntries(new FormData(e.target));
  show(await postJson("/predict/manual", data));
};

qs("#csvForm").onsubmit = async e => {
  e.preventDefault(); setStatus("Processing uploaded CSV files...");
  const fd = new FormData();
  [...e.target.files.files].forEach(f => fd.append("files", f));
  const res = await fetch("/predict/csv", {method:"POST", body:fd});
  show(await res.json());
};

async function postJson(url, data) {
  const res = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)});
  return res.json();
}

function show(data) {
  if (data.error) return setStatus(data.error, true);
  latestResults = data.results;
  setStatus(`${data.results.length} prediction result${data.results.length > 1 ? "s" : ""} generated.`);
  qs("#results").innerHTML = data.results.map((r, i) => `
    <article class="result-card">
      <div class="gauge" style="--p:${r.percent};--c:${r.color}"><strong>${r.percent}%</strong></div>
      <div>
        <span class="risk" style="background:${r.color}">${r.risk}</span>
        <h3>${r.source}${r.source === "Manual input" ? "" : ` - Row ${r.row}`}</h3>
        <p>Confidence probability: <b>${r.probability}</b></p>
        <div class="recs">${r.recommendations.map(x => `<p>${x}</p>`).join("")}</div>
        <button class="primary" onclick="downloadReport(${i})">Download Report</button>
      </div>
    </article>`).join("");
}

async function downloadReport(i) {
  const result = latestResults[i];
  const res = await fetch("/report", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({...patient(), probability:result.probability})});
  const blob = await res.blob(), a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `EEG_AD_Report_${i + 1}.pdf`; a.click(); URL.revokeObjectURL(a.href);
}

function setStatus(text, bad=false) { qs("#status").textContent = text; qs("#status").style.color = bad ? "#b91c1c" : "#1d74b7"; }
