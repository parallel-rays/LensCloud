document.addEventListener('DOMContentLoaded', () => {
    // Grab elements once
    const uploadForm        = document.getElementById('upload-form');
    const uploadSection     = document.querySelector('.upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection    = document.getElementById('results-section');
    const errorSection      = document.getElementById('error-section');
    const progressFill      = document.getElementById('progress-fill');
    const statusMessage     = document.getElementById('status-message');
    const processingTime    = document.getElementById('processing-time');
    const rawVisualized     = document.getElementById('raw_vis-image');
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
    const dropArea          = document.getElementById('drop-area');
    const fileInfo          = document.getElementById('file-info');
    const fileName          = document.getElementById('file-name');
    const fileSize          = document.getElementById('file-size');
    
    currentYear.textContent = new Date().getFullYear();
    
    // Drag & Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('active');
    }
    
    function unhighlight() {
        dropArea.classList.remove('active');
    }
    
    // Handle drop event
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }
    
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            updateFileInput(file);
        }
    }
    
    // Update the file input with the dropped file
    function updateFileInput(file) {
        // Create a new FileList with the dropped file
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInputElm.files = dataTransfer.files;
        
        // Show file info
        fileInfo.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = `(${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        
        // Add preview if it's an image
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Create preview element
                const preview = document.createElement('div');
                preview.className = 'preview-container';
                preview.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" class="file-preview">
                    <div class="preview-overlay">
                        <span>Click to change file or drop new image</span>
                    </div>
                `;
                
                // Replace the drop message with the preview
                const dropMessage = dropArea.querySelector('.drop-message');
                if (dropMessage) {
                    dropArea.removeChild(dropMessage);
                }
                
                // Remove any existing preview
                const existingPreview = dropArea.querySelector('.preview-container');
                if (existingPreview) {
                    dropArea.removeChild(existingPreview);
                }
                
                dropArea.appendChild(preview);
                dropArea.classList.add('has-preview');
            };
            reader.readAsDataURL(file);
        }
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInputElm.dispatchEvent(event);
    }
    
    // Handle regular file input change
    fileInputElm.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            const fileSize = file.size / 1024 / 1024; // size in MB
            
            if (fileSize > 64) {
                showError('File size exceeds the maximum limit of 16MB');
                this.value = '';
                return;
            }
            
            console.log(`File selected: ${file.name} (${fileSize.toFixed(2)} MB)`);
            
            // Update file info display
            updateFileDisplay(file);
        }
    });
    
    // Update the UI to show file info and preview
    function updateFileDisplay(file) {
        fileInfo.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = `(${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        
        // Add preview if it's an image
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Create preview element
                const preview = document.createElement('div');
                preview.className = 'preview-container';
                preview.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" class="file-preview">
                    <div class="preview-overlay">
                        <span>Click to change file or drop new image</span>
                    </div>
                `;
                
                // Replace the drop message with the preview
                const dropMessage = dropArea.querySelector('.drop-message');
                if (dropMessage) {
                    dropArea.removeChild(dropMessage);
                }
                
                // Remove any existing preview
                const existingPreview = dropArea.querySelector('.preview-container');
                if (existingPreview) {
                    dropArea.removeChild(existingPreview);
                }
                
                dropArea.appendChild(preview);
                dropArea.classList.add('has-preview');
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Handle the form SUBMIT
    uploadForm.addEventListener('submit', async e => {
        e.preventDefault();
        
        try {
            // Make sure a file is selected
            if (!fileInputElm.files.length) {
                throw new Error('Please select a file to upload');
            }
            
            // Build a fresh FormData so we can control the field names exactly
            const formData = new FormData();
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
                processingTime.textContent = data.processing_time.toFixed(2);
                
                // Set image sources
                rawVisualized.src = data.raw_vis_url;
                processedImage.src = data.image_url;
                
                // Set download link
                downloadLink.href = data.image_url;
                downloadLink.download = `processed_${fileInputElm.files[0].name.split('.')[0]}.jpg`;
            }, 500);
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        }
    });
    
    // Error handling
    function showError(message) {
        stopProgressSimulation();
        uploadSection.style.display = 'none';
        processingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'block';
        
        errorMessage.textContent = message || 'An error occurred during processing';
    }
    
    // Progress simulation
    let progressInterval;
    let progress = 0;
    
    function startProgressSimulation() {
        progress = 0;
        progressFill.style.width = '0%';
        
        progressInterval = setInterval(() => {
            // Simulate progress that starts fast but slows down towards completion
            if (progress < 90) {
                progress += Math.max(0.5, (100 - progress) / 20);
                progressFill.style.width = `${progress}%`;
                
                // Update status message
                if (progress < 30) {
                    statusMessage.textContent = 'Reading image data...';
                } else if (progress < 60) {
                    statusMessage.textContent = 'Demosaicing Bayer pattern...';
                } else if (progress < 80) {
                    statusMessage.textContent = 'Applying color correction...';
                } else {
                    statusMessage.textContent = 'Finalizing image...';
                }
            }
        }, 100);
    }
    
    function stopProgressSimulation() {
        clearInterval(progressInterval);
    }
    
    // Reset form and start over
    startNewBtn.addEventListener('click', () => {
        resetForm();
    });
    
    tryAgainBtn.addEventListener('click', () => {
        resetForm();
    });
    
    function resetForm() {
        // Reset file input
        fileInputElm.value = '';
        
        // Reset form fields
        document.getElementById('width').value = '';
        document.getElementById('height').value = '';
        document.getElementById('bit-depth').value = '';
        document.getElementById('bayer-pattern').value = '';
        
        // Reset file info
        fileInfo.style.display = 'none';
        
        // Reset drop area
        const existingPreview = dropArea.querySelector('.preview-container');
        if (existingPreview) {
            dropArea.removeChild(existingPreview);
            
            // Add back the drop message if it's not there
            if (!dropArea.querySelector('.drop-message')) {
                const dropMessage = document.createElement('div');
                dropMessage.className = 'drop-message';
                dropMessage.innerHTML = `
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>Drag & drop image here or click to browse</p>
                `;
                dropArea.appendChild(dropMessage);
            }
        }
        
        dropArea.classList.remove('has-preview');
        
        // Show upload section
        uploadSection.style.display = 'block';
        processingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
    }
    
    // API Status check
    apiStatusCheck.addEventListener('click', async (e) => {
        e.preventDefault();
        
        try {
            apiStatus.textContent = 'Checking...';
            apiStatus.className = '';
            
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'online') {
                apiStatus.textContent = 'Online';
                apiStatus.className = 'online';
            } else {
                apiStatus.textContent = 'Offline';
                apiStatus.className = 'offline';
            }
        } catch (error) {
            apiStatus.textContent = 'Offline';
            apiStatus.className = 'offline';
            console.error('API Status check failed:', error);
        }
    });
});