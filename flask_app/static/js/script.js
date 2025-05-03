document.addEventListener('DOMContentLoaded', () => {
    // grab everything once
    const uploadForm        = document.getElementById('upload-form');
    const uploadSection     = document.querySelector('.upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection    = document.getElementById('results-section');
    const errorSection      = document.getElementById('error-section');
    const progressFill      = document.getElementById('progress-fill');
    const statusMessage     = document.getElementById('status-message');
    const processingTime    = document.getElementById('processing-time');
    const origBayerImage    = document.getElementById('processed-image');
    const processedImage    = document.getElementById('processed-image');
    const downloadLink      = document.getElementById('download-link');
    const startNewBtn       = document.getElementById('start-new');
    const tryAgainBtn       = document.getElementById('try-again');
    const errorMessage      = document.getElementById('error-message');
    const apiStatusCheck    = document.getElementById('api-status-check');
    const apiStatus         = document.getElementById('api-status');
    const currentYear       = document.getElementById('current-year');
    const fileInputElm      = document.getElementById('bayer-image');
    const processBtn        = document.getElementById('process-btn');
    currentYear.textContent = new Date().getFullYear();
  
    // Handle the form SUBMIT (not the button click)
    uploadForm.addEventListener('submit', async e => {
      e.preventDefault();
  
      try {
        // Make sure a file is selected
        if (!fileInputElm.files.length) {
          throw new Error('Please select a file to upload');
        }
  
        // Build a fresh FormData so we can control the field names exactly
        const formData = new FormData();
        // <-- this line is critical: match curl's field name
        formData.append('bayer_image', fileInputElm.files[0]);
  
        // Optional metadata
        const width        = document.getElementById('width').value;
        const height       = document.getElementById('height').value;
        const bitDepth     = document.getElementById('bit-depth').value;
        const bayerPattern = document.getElementById('bayer-pattern').value;
        if (width || height || bitDepth || bayerPattern) {
          const md = {};
          if (width)        md.width         = parseInt(width, 10);
          if (height)       md.height        = parseInt(height, 10);
          if (bitDepth)     md.bit_depth     = parseInt(bitDepth, 10);
          if (bayerPattern) md.bayer_pattern = bayerPattern;
          formData.append('metadata', JSON.stringify(md));
        }
  
        // Swap UI
        uploadSection.style.display     = 'none';
        processingSection.style.display = 'block';
        resultsSection.style.display    = 'none';
        errorSection.style.display      = 'none';
        startProgressSimulation();
  
        // Exactly like curl -F …
        const res = await fetch('/process', {
          method: 'POST',
          body: formData
        });
  
        if (!res.ok) {
          const err = await res.json().catch(()=>({error:'Unknown error'}));
          throw new Error(err.error || `Server returned ${res.status}`);
        }
  
        const data = await res.json();
        stopProgressSimulation();
        progressFill.style.width = '100%';
  
        // show results
        setTimeout(() => {
          processingSection.style.display = 'none';
          resultsSection.style.display    = 'block';
          processingTime.textContent      = data.processing_time.toFixed(2);
          processedImage.src              = data.image_url;
          downloadLink.href               = data.image_url;
  
          // give it a nice download name
          const orig = fileInputElm.files[0].name.split('.');
          const ext  = orig.pop();
          downloadLink.download = `${orig.join('.')}_processed.${ext}`;
        }, 500);
  
      } catch (err) {
        stopProgressSimulation();
        showError(err.message);
      }
    });
    
    // Progress simulation variables
    let progressInterval = null;
    
    // Simulate progress animation
    function startProgressSimulation() {
        let progress = 0;
        const simulationSteps = [
            { progress: 15, message: 'Reading raw Bayer data...', delay: 500 },
            { progress: 30, message: 'Analyzing image structure...', delay: 800 },
            { progress: 50, message: 'Initializing deep learning model...', delay: 1000 },
            { progress: 70, message: 'Processing image...', delay: 1500 },
            { progress: 90, message: 'Finalizing results...', delay: 700 }
        ];
        
        let stepIndex = 0;
        
        function updateProgress() {
            if (stepIndex < simulationSteps.length) {
                const step = simulationSteps[stepIndex];
                progress = step.progress;
                progressFill.style.width = progress + '%';
                statusMessage.textContent = step.message;
                stepIndex++;
                
                // Schedule the next update
                setTimeout(updateProgress, step.delay);
            } else {
                // Keep the progress bar moving slightly until the actual process completes
                progressInterval = setInterval(() => {
                    if (progress < 95) {
                        progress += 0.5;
                        progressFill.style.width = progress + '%';
                    }
                }, 300);
            }
        }
        
        // Start the progress simulation
        updateProgress();
    }
    
    function stopProgressSimulation() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }
    
    // Error handling
    function showError(message) {
        processingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'block';
        errorMessage.textContent = message;
        console.error('Error displayed:', message); // Debug log
    }
    
    // Reset to upload form
    startNewBtn.addEventListener('click', () => {
        resetUI();
    });
    
    tryAgainBtn.addEventListener('click', () => {
        resetUI();
    });
    
    function resetUI() {
        uploadForm.reset();
        uploadSection.style.display = 'block';
        processingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
        progressFill.style.width = '0%';
    }
    
    // API status check
    apiStatusCheck.addEventListener('click', async (e) => {
        e.preventDefault();
        
        apiStatus.textContent = ' (checking...)';
        
        try {
            const response = await fetch('/api/test');
            const data = await response.json();
            apiStatus.textContent = ' (online)';
            apiStatus.className = 'online';
        } catch (error) {
            apiStatus.textContent = ' (offline)';
            apiStatus.className = 'offline';
        }
    });
    
    // Check API status on page load
    (async () => {
        try {
            const response = await fetch('/api/test');
            const data = await response.json();
            apiStatus.textContent = ' (online)';
            apiStatus.className = 'online';
            console.log('response')
            console.log(response)
        } catch (error) {
            apiStatus.textContent = ' (offline)';
            apiStatus.className = 'offline';
        }
    })();
    
    // File input styling and validation
    const fileInput = document.getElementById('bayer-image');
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const fileName = this.files[0].name;
            const fileSize = this.files[0].size / 1024 / 1024; // size in MB
            
            if (fileSize > 16) {
                showError('File size exceeds the maximum limit of 16MB');
                this.value = '';
                return;
            }
            
            console.log(`File selected: ${fileName} (${fileSize.toFixed(2)} MB)`); // Debug log
            
            // Visual feedback that file is selected
            this.style.borderColor = 'var(--success-color)';
            this.style.borderStyle = 'solid';
        } else {
            this.style.borderColor = 'var(--border-color)';
            this.style.borderStyle = 'dashed';
        }
    });
    
    // Add form submission debugging
    processBtn.addEventListener('click', () => {
        console.log('Process button clicked'); // Debug log
    });
    
    console.log('JavaScript initialized - DOM fully loaded'); // Debug log
});