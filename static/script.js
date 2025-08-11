document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const formData = new FormData(e.target);
  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = isNaN(value) ? value : Number(value);
  });

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  const resultDiv = document.getElementById("result");
  resultDiv.classList.remove("d-none", "alert-success", "alert-danger");

  if (data.attrition === "Yes") {
    resultDiv.classList.add("alert-danger");
    resultDiv.innerHTML = `⚠️ <strong>High Attrition Risk!</strong><br>Probability: ${(data.probability*100).toFixed(1)}%`;
  } else {
    resultDiv.classList.add("alert-success");
    resultDiv.innerHTML = `✅ <strong>Low Attrition Risk</strong><br>Probability: ${(data.probability*100).toFixed(1)}%`;
  }

  resultDiv.scrollIntoView({behavior: "smooth"});
});
