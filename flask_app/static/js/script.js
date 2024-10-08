// Get the prediction element
const predictionElement = document.querySelector('.prediction');

// Create a function to handle the upload form submission
function handleFormSubmit(event) {
    event.preventDefault();

    // Get the uploaded file
    const file = event.target.elements['file'].files[0];

    // Create a new FormData object
    const formData = new FormData();

    // Add the uploaded file to the FormData object
    formData.append('file', file);

    // Make a POST request to the `/upload` endpoint
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update the prediction element with the prediction
        predictionElement.textContent = `Prediction: ${data.prediction}`;
    })
    .catch(error => {
        // Handle the error
        console.error(error);
    });
}

// Add an event listener to the form submission event
document.querySelector('form').addEventListener('submit', handleFormSubmit);
