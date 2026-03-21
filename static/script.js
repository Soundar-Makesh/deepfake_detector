document.addEventListener('DOMContentLoaded', () => {
    // Initialize icons
    lucide.createIcons();

    const form = document.getElementById('uploadForm');
    const input = document.getElementById('videoInput');
    const dropZone = document.getElementById('dropZone');
    const scanner = document.getElementById('scanner');
    const resultBox = document.getElementById('resultBox');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadText = document.getElementById('uploadText');
    const uploadIcon = document.getElementById('uploadIcon');

    // Drag and Drop Effects
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('drop-zone-active');
    }

    function unhighlight() {
        dropZone.classList.remove('drop-zone-active');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        input.files = files; // Assign files to input
        updateFileName();
    }

    input.addEventListener('change', updateFileName);

    function updateFileName() {
        if (input.files.length > 0) {
            const file = input.files[0];
            uploadText.innerText = file.name;
            uploadText.style.color = "#E2E8F0"; // slate-200
            
            // Change icon to video
            uploadIcon.setAttribute('data-lucide', 'video');
            lucide.createIcons();
            uploadIcon.classList.add('text-primary');
        }
    }

    form.onsubmit = async (e) => {
        e.preventDefault();
        const file = input.files[0];
        if (!file) {
            alert("Please select or drop a video file first.");
            return;
        }

        const startTime = performance.now();
        
        // UI Loading State
        analyzeBtn.disabled = true;
        document.getElementById('btnText').innerText = "EXTRACTING FORENSICS...";
        document.getElementById('btnIcon').setAttribute('data-lucide', 'loader-2');
        document.getElementById('btnIcon').classList.add('animate-spin');
        lucide.createIcons();
        
        scanner.classList.add('scanner-active');
        
        // Hide previous results smoothly
        resultBox.style.opacity = '0';
        setTimeout(() => resultBox.classList.add('hidden'), 500);

        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server analysis failed");
            }
            
            const data = await response.json();
            
            const endTime = performance.now();
            document.getElementById('latency').innerText = `${Math.round(endTime - startTime)}ms`;

            if (data.error) {
                throw new Error(data.error);
            } else {
                updateUI(data);
            }
        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
        } finally {
            // Restore UI
            analyzeBtn.disabled = false;
            document.getElementById('btnText').innerText = "INITIATE ANALYSIS";
            document.getElementById('btnIcon').setAttribute('data-lucide', 'scan-line');
            document.getElementById('btnIcon').classList.remove('animate-spin');
            lucide.createIcons();
            scanner.classList.remove('scanner-active');
        }
    };

    function updateUI(data) {
        // Show result box smoothly
        resultBox.classList.remove('hidden');
        requestAnimationFrame(() => {
            resultBox.style.opacity = '1';
        });

        const verdict = document.getElementById('verdictText');
        const conf = document.getElementById('confidenceText');
        const bar = document.getElementById('confidenceBar');
        const iconBox = document.getElementById('verdictIconBox');
        const vIcon = document.getElementById('verdictIcon');

        verdict.innerText = data.prediction;
        
        // Animate counter
        animateValue(conf, 0, Math.round(data.confidence), 1000);

        // Determination of colors and icons based on 3-tier result
        let themeColor, bgIconColor, iconName;
        
        if (data.prediction === 'FAKE') {
            themeColor = "#F43F5E"; // Pink-red
            bgIconColor = "rgba(244, 63, 94, 0.15)";
            iconName = 'alert-octagon';
        } else if (data.prediction === 'REAL') {
            themeColor = "#10B981"; // Emerald green
            bgIconColor = "rgba(16, 185, 129, 0.15)";
            iconName = 'shield-check';
        } else {
            themeColor = "#F59E0B"; // Amber / Yellow
            bgIconColor = "rgba(245, 158, 11, 0.15)";
            iconName = 'help-circle';
        }
        
        verdict.style.color = themeColor;
        conf.style.color = themeColor;
        bar.style.backgroundColor = themeColor;
        iconBox.style.backgroundColor = bgIconColor;
        vIcon.style.color = themeColor;

        vIcon.setAttribute('data-lucide', iconName);
        lucide.createIcons();

        // Animate bar
        setTimeout(() => {
            bar.style.width = `${data.confidence}%`;
        }, 100);
    }
    
    // Smooth number counting utility
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            obj.innerHTML = Math.floor(easeOut * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end;
            }
        };
        window.requestAnimationFrame(step);
    }
});