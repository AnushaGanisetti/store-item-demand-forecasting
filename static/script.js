
// Get HTML elements
const storeSelect = document.getElementById("store");
const itemSelect = document.getElementById("item");
const dateInput = document.getElementById("date");
const resultDiv = document.getElementById("result");

// Populate Store dropdown (1 to 10)
for (let i = 1; i <= 10; i++) {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = "Store " + i;
    storeSelect.appendChild(option);
}

// Populate Item dropdown (1 to 50)
for (let i = 1; i <= 50; i++) {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = "Item " + i;
    itemSelect.appendChild(option);
}

// Function to predict sales
async function predictSales() {

    const date = dateInput.value;
    const store = storeSelect.value;
    const item = itemSelect.value;

    // Validate inputs
    if (!date || !store || !item) {
        resultDiv.style.color = "red";
        resultDiv.textContent =
            "Please select date, store and item.";
        return;
    }

    // Display loading message
    resultDiv.style.color = "#2563eb";
    resultDiv.textContent = "Predicting sales...";

    try {

        const url =
            `/api/forecast?date=${encodeURIComponent(date)}&store=${store}&item=${item}`;

        const response = await fetch(url);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        // Display prediction
        resultDiv.style.color = "#15803d";

        resultDiv.innerHTML = `
            <h3>Prediction Result</h3>
            <p>Predicted Sales: ${Math.round(data.predicted_sales)} units</p>
            <p>Exact Prediction: ${data.predicted_sales} units</p>
            <p>Date: ${data.date}</p>
            <p>Store: ${data.store}</p>
            <p>Item: ${data.item}</p>
        `;

    } catch (error) {

        resultDiv.style.color = "red";
        resultDiv.textContent = error.message;

    }
}
