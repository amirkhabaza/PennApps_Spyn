const { ipcRenderer } = require('electron');

// ===== MAIN DASHBOARD CLASS =====
class Spyn {
    constructor() {
        this.isMonitoring = false;
        this.sessionStartTime = null;
        this.sessionTimer = null;
        this.postureData = [];
        this.isGoodPosture = true;
        this.overlayMode = 'status';
        this.transparencyLevel = 'visible';
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        
        this.initializeElements();
        this.bindEvents();
        this.initializeChart();
        this.setupIPC();
    }

    initializeElements() {
        // Dashboard elements
        this.startBtn = document.getElementById('startMonitoringBtn');
        this.postureReport = document.getElementById('postureReport');
        
        // Exercise elements
        this.startExerciseBtn = document.getElementById('startExerciseBtn');
        this.exerciseAnalysis = document.getElementById('exerciseAnalysis');
        
        // Modal elements
        this.shortcutsBtn = document.getElementById('shortcutsBtn');
        this.shortcutsModal = document.getElementById('shortcutsModal');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        
        // Sign-out element
        this.signoutBtn = document.getElementById('signoutBtn');
        
        // Report elements
        this.overallScore = document.getElementById('overallScore');
        this.sessionDuration = document.getElementById('sessionDuration');
        this.goodPostureTime = document.getElementById('goodPostureTime');
        this.corrections = document.getElementById('corrections');
    }

    bindEvents() {
        // Dashboard events
        if (this.startBtn) {
            this.startBtn.addEventListener('click', () => this.toggleMonitoring());
        }
        
        // Exercise events
        if (this.startExerciseBtn) {
            this.startExerciseBtn.addEventListener('click', () => this.startExerciseAnalysis());
        }
        
        // Tab switching events
        this.initializeTabs();
        
        // Modal events
        if (this.shortcutsBtn) {
            this.shortcutsBtn.addEventListener('click', () => this.showShortcutsModal());
        }
        if (this.closeModalBtn) {
            this.closeModalBtn.addEventListener('click', () => this.hideShortcutsModal());
        }
        if (this.shortcutsModal) {
            this.shortcutsModal.addEventListener('click', (e) => {
                if (e.target === this.shortcutsModal) {
                    this.hideShortcutsModal();
                }
            });
        }
        if (this.signoutBtn) {
            this.signoutBtn.addEventListener('click', () => this.handleSignOut());
        }
        
        // Keyboard shortcuts
        this.initializeKeyboardShortcuts();
    }

    initializeTabs() {
        // Get tab buttons and panes
        this.tabButtons = document.querySelectorAll('.tab-btn');
        this.tabPanes = document.querySelectorAll('.tab-pane');
        
        // Add click event listeners to tab buttons
        this.tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const targetTab = e.target.getAttribute('data-tab');
                this.switchTab(targetTab);
            });
        });
    }

    switchTab(tabName) {
        // Remove active class from all tabs and panes
        this.tabButtons.forEach(btn => btn.classList.remove('active'));
        this.tabPanes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to selected tab and pane
        const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
        const activePane = document.getElementById(`${tabName}-tab`);
        
        if (activeButton && activePane) {
            activeButton.classList.add('active');
            activePane.classList.add('active');
            
            // If switching to exercise tab, close the overlay
            if (tabName === 'exercise' && this.isMonitoring) {
                this.stopMonitoring();
            }
        }
    }

    setupIPC() {
        // Listen for messages from main process
        ipcRenderer.on('overlay-closed', () => {
            this.stopMonitoring();
        });
    }

    toggleMonitoring() {
        if (this.isMonitoring) {
            this.stopMonitoring();
        } else {
            this.startMonitoring();
        }
    }

    startMonitoring() {
        this.isMonitoring = true;
        this.sessionStartTime = Date.now();
        this.postureData = [];
        
        // Update UI
        if (this.startBtn) {
            this.startBtn.innerHTML = '<span class="btn-icon">⏹</span>Stop Monitoring';
            this.startBtn.style.background = 'linear-gradient(135deg, #ff4444, #cc0000)';
            this.startBtn.style.boxShadow = '0 8px 32px rgba(255, 68, 68, 0.3)';
        }
        
        // Hide report
        if (this.postureReport) {
            this.postureReport.classList.add('hidden');
        }
        
        // Start timer
        this.startTimer();
        
        // Simulate posture monitoring (replace with actual AI logic later)
        this.startPostureSimulation();
        
        // Notify main process to start monitoring
        ipcRenderer.invoke('start-monitoring');
        
        console.log('Spyn monitoring started');
    }

    stopMonitoring() {
        this.isMonitoring = false;
        
        // Update UI
        if (this.startBtn) {
            this.startBtn.innerHTML = '<span class="btn-icon">▶</span>Start Spyn Monitoring';
            this.startBtn.style.background = 'linear-gradient(135deg, #00bfff, #0099cc)';
            this.startBtn.style.boxShadow = '0 8px 32px rgba(0, 191, 255, 0.3)';
        }
        
        // Show report
        this.showPostureReport();
        
        // Stop timer
        clearInterval(this.timerInterval);
        
        // Notify main process to stop monitoring
        ipcRenderer.invoke('stop-monitoring');
        
        console.log('Spyn monitoring stopped');
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            if (this.sessionStartTime) {
                const elapsed = Date.now() - this.sessionStartTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                if (this.sessionTimer) {
                    this.sessionTimer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                }
            }
        }, 1000);
    }

    startPostureSimulation() {
        // Simulate posture changes every 3-8 seconds
        this.postureSimulation = setInterval(() => {
            if (this.isMonitoring) {
                this.simulatePostureChange();
            }
        }, Math.random() * 5000 + 3000);
    }

    simulatePostureChange() {
        // Randomly change posture status
        const wasGoodPosture = this.isGoodPosture;
        this.isGoodPosture = Math.random() > 0.3; // 70% chance of good posture
        
        // Record posture data
        this.postureData.push({
            timestamp: Date.now(),
            isGood: this.isGoodPosture,
            percentage: this.calculateGoodPosturePercentage()
        });
        
        console.log(`Posture changed: ${this.isGoodPosture ? 'Good' : 'Bad'}`);
    }

    calculateGoodPosturePercentage() {
        if (this.postureData.length === 0) return 100;
        
        const goodCount = this.postureData.filter(data => data.isGood).length;
        return Math.round((goodCount / this.postureData.length) * 100);
    }

    showPostureReport() {
        // Calculate session metrics
        const sessionDuration = this.sessionStartTime ? 
            Math.floor((Date.now() - this.sessionStartTime) / 1000) : 0;
        
        const minutes = Math.floor(sessionDuration / 60);
        const seconds = sessionDuration % 60;
        
        const goodPosturePercentage = this.calculateGoodPosturePercentage();
        const corrections = this.postureData.filter((data, index) => 
            index > 0 && data.isGood !== this.postureData[index - 1].isGood
        ).length;
        
        // Update report metrics
        if (this.overallScore) this.overallScore.textContent = `${goodPosturePercentage}%`;
        if (this.sessionDuration) this.sessionDuration.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        if (this.goodPostureTime) this.goodPostureTime.textContent = `${goodPosturePercentage}%`;
        if (this.corrections) this.corrections.textContent = corrections.toString();
        
        // Show report
        if (this.postureReport) {
            this.postureReport.classList.remove('hidden');
        }
        
        // Update chart
        this.updateChart();
    }

    initializeChart() {
        this.chartCanvas = document.getElementById('postureChart');
        if (this.chartCanvas) {
            this.chartCtx = this.chartCanvas.getContext('2d');
        }
    }

    updateChart() {
        if (!this.chartCtx || this.postureData.length === 0) return;
        
        const canvas = this.chartCanvas;
        const ctx = this.chartCtx;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Set up chart dimensions
        const padding = 40;
        const chartWidth = canvas.width - (padding * 2);
        const chartHeight = canvas.height - (padding * 2);
        
        // Draw background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(padding, padding, chartWidth, chartHeight);
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(padding + chartWidth, y);
            ctx.stroke();
        }
        
        // Draw posture data
        if (this.postureData.length > 1) {
            const pointSpacing = chartWidth / (this.postureData.length - 1);
            
            ctx.strokeStyle = '#00bfff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            this.postureData.forEach((data, index) => {
                const x = padding + (pointSpacing * index);
                const y = padding + chartHeight - (chartHeight * (data.percentage / 100));
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Draw data points
            ctx.fillStyle = '#00bfff';
            this.postureData.forEach((data, index) => {
                const x = padding + (pointSpacing * index);
                const y = padding + chartHeight - (chartHeight * (data.percentage / 100));
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
        }
        
        // Draw labels
        ctx.fillStyle = '#888';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        
        // Y-axis labels
        for (let i = 0; i <= 4; i++) {
            const y = padding + (chartHeight / 4) * i;
            const value = 100 - (i * 25);
            ctx.fillText(`${value}%`, padding - 10, y + 4);
        }
    }

    startExerciseAnalysis() {
        // Show the exercise analysis section
        if (this.exerciseAnalysis) {
            this.exerciseAnalysis.classList.remove('hidden');
            
            // Update button text
            if (this.startExerciseBtn) {
                this.startExerciseBtn.textContent = '📹 Exercise Analysis Active';
                this.startExerciseBtn.disabled = true;
            }
            
            // Start exercise simulation
            this.startExerciseSimulation();
            
            console.log('Exercise analysis started');
        }
    }

    startExerciseSimulation() {
        // Simulate exercise data updates
        this.exerciseInterval = setInterval(() => {
            // Update form score (simulate slight variations)
            const formScore = document.getElementById('formScore');
            if (formScore) {
                const currentScore = parseInt(formScore.textContent);
                const variation = Math.floor(Math.random() * 6) - 3; // -3 to +3
                const newScore = Math.max(80, Math.min(100, currentScore + variation));
                formScore.textContent = `${newScore}%`;
            }
            
            // Update reps count
            const repsCount = document.getElementById('repsCount');
            if (repsCount) {
                const currentReps = parseInt(repsCount.textContent);
                repsCount.textContent = currentReps + 1;
            }
            
            // Update corrections made
            const correctionsMade = document.getElementById('correctionsMade');
            if (correctionsMade && Math.random() > 0.7) { // 30% chance of correction
                const currentCorrections = parseInt(correctionsMade.textContent);
                correctionsMade.textContent = currentCorrections + 1;
            }
        }, 3000); // Update every 3 seconds
    }

    initializeKeyboardShortcuts() {
        // Add keyboard event listener to document
        document.addEventListener('keydown', (e) => {
            // Prevent shortcuts from triggering in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }

            // Handle shortcuts based on key combinations
            if (e.ctrlKey || e.metaKey) {
                switch (e.key.toLowerCase()) {
                    case 'm':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.toggleMonitoring();
                        }
                        break;
                    case 'p':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.toggleOverlay();
                        }
                        break;
                    case 't':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.setTransparency('transparent');
                        }
                        break;
                    case 'f':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.setTransparency('visible');
                        }
                        break;
                    case 'c':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.toggleCamera();
                        }
                        break;
                    case 'o':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.centerOverlay();
                        }
                        break;
                }
            }

            // Function key shortcuts
            switch (e.key) {
                case 'F9':
                    e.preventDefault();
                    this.toggleMonitoring();
                    break;
                case 'F10':
                    e.preventDefault();
                    this.toggleOverlay();
                    break;
                case 'F11':
                    e.preventDefault();
                    this.toggleCamera();
                    break;
                case 'F1':
                    e.preventDefault();
                    this.setTransparency('visible');
                    break;
                case 'F2':
                    e.preventDefault();
                    this.setTransparency('transparent');
                    break;
                case 'F3':
                    e.preventDefault();
                    this.switchTab('posture');
                    break;
                case 'F4':
                    e.preventDefault();
                    this.switchTab('exercise');
                    break;
                case 'Escape':
                    e.preventDefault();
                    if (this.shortcutsModal && !this.shortcutsModal.classList.contains('hidden')) {
                        this.hideShortcutsModal();
                    } else if (this.isMonitoring) {
                        this.stopMonitoring();
                    }
                    break;
                case 'F12':
                    e.preventDefault();
                    this.toggleShortcutsModal();
                    break;
            }
        });

        console.log('Keyboard shortcuts initialized');
    }

    toggleOverlay() {
        ipcRenderer.invoke('toggle-overlay');
    }

    setTransparency(level) {
        ipcRenderer.invoke('set-transparency', level);
    }

    toggleCamera() {
        ipcRenderer.invoke('show-camera');
    }

    centerOverlay() {
        ipcRenderer.invoke('center-overlay');
    }

    showShortcutsModal() {
        if (this.shortcutsModal) {
            this.shortcutsModal.classList.remove('hidden');
        }
    }

    hideShortcutsModal() {
        if (this.shortcutsModal) {
            this.shortcutsModal.classList.add('hidden');
        }
    }

    toggleShortcutsModal() {
        if (this.shortcutsModal) {
            if (this.shortcutsModal.classList.contains('hidden')) {
                this.showShortcutsModal();
            } else {
                this.hideShortcutsModal();
            }
        }
    }

    handleSignOut() {
        // Stop monitoring if active
        if (this.isMonitoring) {
            this.stopMonitoring();
        }
        
        // Show confirmation message
        const confirmation = confirm('Are you sure you want to sign out?');
        if (confirmation) {
            // Navigate to sign-in page
            ipcRenderer.send('navigate-to-signin');
        }
    }
}

// ===== CAMERA MANAGER CLASS =====
class CameraManager {
    constructor() {
        this.videoElement = document.getElementById('videoElement');
        this.loadingMessage = document.getElementById('loadingMessage');
        this.errorMessage = document.getElementById('errorMessage');
        this.cameraStatus = document.getElementById('cameraStatus');
        this.closeBtn = document.getElementById('closeBtn');
        
        this.stream = null;
        this.isCameraActive = false;
        
        this.initializeCamera();
        this.bindEvents();
    }

    bindEvents() {
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => {
                // Just hide the camera window, don't stop the camera or close the window
                ipcRenderer.invoke('hide-camera');
            });
        }

        // Handle window close (when user actually closes the window)
        window.addEventListener('beforeunload', () => {
            this.stopCamera();
        });
    }

    async initializeCamera() {
        try {
            this.showLoading('Requesting camera access...');
            
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' // Front camera
                },
                audio: false
            });

            // Set video source
            this.videoElement.srcObject = this.stream;
            
            // Wait for video to load
            this.videoElement.onloadedmetadata = () => {
                this.hideLoading();
                this.showCamera();
                this.isCameraActive = true;
                this.cameraStatus.textContent = 'Camera Active';
                
                console.log('Camera initialized successfully');
            };

            this.videoElement.onerror = (error) => {
                console.error('Video error:', error);
                this.showError('Failed to load camera stream');
            };

        } catch (error) {
            console.error('Camera access error:', error);
            this.handleCameraError(error);
        }
    }

    showLoading(message) {
        if (this.loadingMessage) {
            this.loadingMessage.textContent = message;
            this.loadingMessage.style.display = 'block';
        }
        if (this.errorMessage) {
            this.errorMessage.style.display = 'none';
        }
        if (this.videoElement) {
            this.videoElement.style.display = 'none';
        }
    }

    hideLoading() {
        if (this.loadingMessage) {
            this.loadingMessage.style.display = 'none';
        }
    }

    showCamera() {
        if (this.videoElement) {
            this.videoElement.style.display = 'block';
        }
        if (this.errorMessage) {
            this.errorMessage.style.display = 'none';
        }
    }

    showError(message) {
        if (this.errorMessage) {
            this.errorMessage.textContent = message;
            this.errorMessage.style.display = 'block';
        }
        if (this.loadingMessage) {
            this.loadingMessage.style.display = 'none';
        }
        if (this.videoElement) {
            this.videoElement.style.display = 'none';
        }
        if (this.cameraStatus) {
            this.cameraStatus.textContent = 'Camera Error';
            this.cameraStatus.className = 'camera-status camera-error';
        }
    }

    handleCameraError(error) {
        let errorMessage = 'Camera access denied';
        
        if (error.name === 'NotAllowedError') {
            errorMessage = 'Camera access denied. Please allow camera access and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No camera found. Please connect a camera and try again.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Camera is being used by another application.';
        } else if (error.name === 'OverconstrainedError') {
            errorMessage = 'Camera constraints cannot be satisfied.';
        } else {
            errorMessage = `Camera error: ${error.message}`;
        }
        
        this.showError(errorMessage);
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
            this.stream = null;
        }
        
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        
        this.isCameraActive = false;
        console.log('Camera stopped');
    }
}

// ===== OVERLAY CLASS =====
class SpynOverlay {
    constructor() {
        this.isMonitoring = false;
        this.sessionStartTime = null;
        this.timerInterval = null;
        this.postureData = [];
        this.isGoodPosture = true;
        this.cameraVisible = false;
        this.transparencyLevel = 'visible';
        
        this.initializeElements();
        this.bindEvents();
        this.setupIPC();
        
        // Start timer after a short delay to ensure DOM is ready
        setTimeout(() => {
            this.startTimer();
        }, 100);
    }

    initializeElements() {
        // Overlay elements
        this.sessionTimer = document.getElementById('sessionTimer');
        this.postureStatus = document.getElementById('postureStatus');
        this.cameraToggleBtn = document.getElementById('cameraToggleBtn');
        this.posturePercentage = document.getElementById('posturePercentage');
        
        // Transparency controls
        this.transparentBtn = document.getElementById('transparentBtn');
        this.visibleBtn = document.getElementById('visibleBtn');
        
        // Container for state management
        this.container = document.querySelector('.overlay-container');
        
        // Debug element initialization
        console.log('Elements initialized:', {
            sessionTimer: !!this.sessionTimer,
            postureStatus: !!this.postureStatus,
            cameraToggleBtn: !!this.cameraToggleBtn
        });
    }

    bindEvents() {
        // Camera toggle button
        if (this.cameraToggleBtn) {
            this.cameraToggleBtn.addEventListener('click', () => this.toggleCamera());
        }
        
        // Transparency controls
        if (this.transparentBtn) {
            this.transparentBtn.addEventListener('click', () => this.setTransparency('transparent'));
        }
        if (this.visibleBtn) {
            this.visibleBtn.addEventListener('click', () => this.setTransparency('visible'));
        }
    }

    setupIPC() {
        // Listen for messages from main process
        ipcRenderer.on('start-monitoring', () => {
            this.startMonitoring();
        });

        ipcRenderer.on('stop-monitoring', () => {
            this.stopMonitoring();
        });

        ipcRenderer.on('transparency-changed', (event, level) => {
            if (this.container) {
                this.container.classList.remove('transparent', 'visible');
                this.container.classList.add(level);
            }
        });
    }

    startMonitoring() {
        this.isMonitoring = true;
        this.sessionStartTime = Date.now();
        this.postureData = [];
        
        // Camera starts hidden by default
        this.cameraVisible = false;
        this.updateCameraToggleButton();
        
        // Restart timer with new start time
        this.startTimer();
        
        // Start posture simulation
        this.startPostureSimulation();
        
        console.log('Overlay: Monitoring started');
    }

    stopMonitoring() {
        this.isMonitoring = false;
        
        // Stop timer
        clearInterval(this.timerInterval);
        
        // Stop posture simulation
        if (this.postureSimulation) {
            clearInterval(this.postureSimulation);
        }
        
        console.log('Overlay: Monitoring stopped');
        
        // Close the overlay window
        setTimeout(() => {
            window.close();
        }, 1000); // Give a brief moment to show final stats
    }

    startTimer() {
        // Clear any existing timer
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        // Ensure we have a start time
        if (!this.sessionStartTime) {
            this.sessionStartTime = Date.now();
        }
        
        // Start the timer immediately
        this.updateTimer();
        
        this.timerInterval = setInterval(() => {
            this.updateTimer();
        }, 1000);
        
        console.log('Timer started with start time:', this.sessionStartTime);
    }
    
    updateTimer() {
        if (this.sessionStartTime) {
            const elapsed = Date.now() - this.sessionStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Re-find elements in case they weren't available initially
            if (!this.sessionTimer) {
                this.sessionTimer = document.getElementById('sessionTimer');
            }
            
            // Update timer
            if (this.sessionTimer) {
                this.sessionTimer.textContent = timeString;
            }
            
            console.log('Timer updated:', timeString);
        }
    }

    startPostureSimulation() {
        // Simulate posture changes every 3-8 seconds
        this.postureSimulation = setInterval(() => {
            if (this.isMonitoring) {
                this.simulatePostureChange();
            }
        }, Math.random() * 5000 + 3000);
    }

    simulatePostureChange() {
        // Randomly change posture status
        const wasGoodPosture = this.isGoodPosture;
        this.isGoodPosture = Math.random() > 0.3; // 70% chance of good posture
        
        // Record posture data
        this.postureData.push({
            timestamp: Date.now(),
            isGood: this.isGoodPosture,
            percentage: this.calculateGoodPosturePercentage()
        });
        
        this.updatePostureStatus();
        
        // Handle bad posture behavior
        if (!this.isGoodPosture && wasGoodPosture) {
            this.handleBadPosture();
        } else if (this.isGoodPosture && !wasGoodPosture) {
            this.handleGoodPosture();
        }
        
        this.updatePosturePercentage();
    }

    updatePostureStatus() {
        const statusElement = this.postureStatus;
        if (!statusElement) return;
        
        const statusIcon = statusElement.querySelector('.status-icon');
        const statusText = statusElement.querySelector('.status-text');
        
        if (this.isGoodPosture) {
            statusElement.className = 'posture-status good';
            if (statusIcon) statusIcon.textContent = '✓';
            if (statusText) statusText.textContent = 'Good';
        } else {
            statusElement.className = 'posture-status bad';
            if (statusIcon) statusIcon.textContent = '✗';
            if (statusText) statusText.textContent = 'Bad';
        }
    }

    handleBadPosture() {
        // Add visual indicator for bad posture without moving the window
        document.body.classList.add('bad-posture');
        
        console.log('Bad posture detected - visual indicator shown');
    }

    handleGoodPosture() {
        // Remove visual indicator for bad posture
        document.body.classList.remove('bad-posture');
        
        console.log('Good posture restored - visual indicator removed');
    }

    calculateGoodPosturePercentage() {
        if (this.postureData.length === 0) return 100;
        
        const goodCount = this.postureData.filter(data => data.isGood).length;
        return Math.round((goodCount / this.postureData.length) * 100);
    }

    updatePosturePercentage() {
        const percentage = this.calculateGoodPosturePercentage();
        if (this.posturePercentage) {
            this.posturePercentage.textContent = `${percentage}%`;
        }
    }

    toggleCamera() {
        this.cameraVisible = !this.cameraVisible;
        
        if (this.cameraVisible) {
            console.log('Camera toggled ON');
            this.showCamera();
        } else {
            console.log('Camera toggled OFF');
            this.hideCamera();
        }
        
        this.updateCameraToggleButton();
    }

    updateCameraToggleButton() {
        if (this.cameraToggleBtn) {
            if (this.cameraVisible) {
                this.cameraToggleBtn.classList.add('active');
                const cameraText = this.cameraToggleBtn.querySelector('.camera-text');
                if (cameraText) cameraText.textContent = 'CAMERA ON';
            } else {
                this.cameraToggleBtn.classList.remove('active');
                const cameraText = this.cameraToggleBtn.querySelector('.camera-text');
                if (cameraText) cameraText.textContent = 'CAMERA OFF';
            }
        }
    }

    showCamera() {
        // Notify main process to show camera window (will show existing hidden window)
        ipcRenderer.invoke('show-camera');
        console.log('Camera window show requested');
    }

    hideCamera() {
        // Notify main process to hide camera window (will hide but keep running)
        ipcRenderer.invoke('hide-camera');
        console.log('Camera window hide requested');
    }

    setTransparency(level) {
        this.transparencyLevel = level;
        
        // Update button states
        document.querySelectorAll('.transparency-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        const activeBtn = document.querySelector(`[data-level="${level}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
        
        // Notify main process to handle transparency
        ipcRenderer.invoke('set-transparency', level);
        
        console.log(`Transparency set to: ${level}`);
    }
}

// ===== SIGN-IN PAGE CLASS =====
class SignInPage {
    constructor() {
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Form elements
        this.signinForm = document.getElementById('signinForm');
        this.emailInput = document.getElementById('email');
        this.passwordInput = document.getElementById('password');
        this.passwordToggle = document.getElementById('passwordToggle');
        this.rememberMeCheckbox = document.getElementById('rememberMe');
        this.signinBtn = document.getElementById('signinBtn');
        this.btnText = this.signinBtn ? this.signinBtn.querySelector('.btn-text') : null;
        this.btnLoader = this.signinBtn ? this.signinBtn.querySelector('.btn-loader') : null;
        
        // Error elements
        this.emailError = document.getElementById('emailError');
        this.passwordError = document.getElementById('passwordError');
        
        // Social buttons
        this.googleBtn = document.querySelector('.google-btn');
        this.appleBtn = document.querySelector('.apple-btn');
        
        // Demo button
        this.demoBtn = document.getElementById('demoBtn');
        
        // Links
        this.signupLink = document.getElementById('signupLink');
        this.forgotPasswordLink = document.querySelector('.forgot-password');
    }

    bindEvents() {
        // Form submission
        if (this.signinForm) {
            this.signinForm.addEventListener('submit', (e) => this.handleSignIn(e));
        }
        
        // Password toggle
        if (this.passwordToggle) {
            this.passwordToggle.addEventListener('click', () => this.togglePasswordVisibility());
        }
        
        // Input validation
        if (this.emailInput) {
            this.emailInput.addEventListener('blur', () => this.validateEmail());
        }
        if (this.passwordInput) {
            this.passwordInput.addEventListener('blur', () => this.validatePassword());
        }
        
        // Social sign in
        if (this.googleBtn) {
            this.googleBtn.addEventListener('click', () => this.handleGoogleSignIn());
        }
        if (this.appleBtn) {
            this.appleBtn.addEventListener('click', () => this.handleAppleSignIn());
        }
        
        // Demo access
        if (this.demoBtn) {
            this.demoBtn.addEventListener('click', () => this.handleDemoAccess());
        }
        
        // Links
        if (this.signupLink) {
            this.signupLink.addEventListener('click', (e) => this.handleSignupLink(e));
        }
        if (this.forgotPasswordLink) {
            this.forgotPasswordLink.addEventListener('click', (e) => this.handleForgotPassword(e));
        }
        
        // Enter key handling
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.signinBtn && !this.signinBtn.disabled) {
                this.handleSignIn(e);
            }
        });
    }

    handleSignIn(e) {
        e.preventDefault();
        
        // Validate form
        const isEmailValid = this.validateEmail();
        const isPasswordValid = this.validatePassword();
        
        if (!isEmailValid || !isPasswordValid) {
            return;
        }
        
        // Show loading state
        this.setLoadingState(true);
        
        // Simulate API call
        setTimeout(() => {
            this.setLoadingState(false);
            this.showSuccessMessage();
            
            // Navigate to main app after a short delay
            setTimeout(() => {
                this.navigateToMainApp();
            }, 1500);
        }, 2000);
    }

    validateEmail() {
        const email = this.emailInput.value.trim();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        
        if (!email) {
            this.showError(this.emailError, 'Email is required');
            return false;
        } else if (!emailRegex.test(email)) {
            this.showError(this.emailError, 'Please enter a valid email address');
            return false;
        } else {
            this.clearError(this.emailError);
            return true;
        }
    }

    validatePassword() {
        const password = this.passwordInput.value;
        
        if (!password) {
            this.showError(this.passwordError, 'Password is required');
            return false;
        } else if (password.length < 6) {
            this.showError(this.passwordError, 'Password must be at least 6 characters');
            return false;
        } else {
            this.clearError(this.passwordError);
            return true;
        }
    }

    showError(errorElement, message) {
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
    }

    clearError(errorElement) {
        if (errorElement) {
            errorElement.textContent = '';
            errorElement.style.display = 'none';
        }
    }

    setLoadingState(isLoading) {
        if (this.signinBtn) {
            this.signinBtn.disabled = isLoading;
            if (isLoading) {
                this.signinBtn.classList.add('loading');
                if (this.btnText) this.btnText.classList.add('hidden');
                if (this.btnLoader) this.btnLoader.classList.remove('hidden');
            } else {
                this.signinBtn.classList.remove('loading');
                if (this.btnText) this.btnText.classList.remove('hidden');
                if (this.btnLoader) this.btnLoader.classList.add('hidden');
            }
        }
    }

    showSuccessMessage() {
        // Create success message
        const successMessage = document.createElement('div');
        successMessage.className = 'success-message';
        successMessage.innerHTML = `
            <div class="success-icon">✓</div>
            <div class="success-text">Welcome back! Redirecting to dashboard...</div>
        `;
        
        // Add styles
        successMessage.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #00bfff, #0099cc);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 191, 255, 0.3);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 1rem;
            animation: slideIn 0.5s ease-out;
        `;
        
        // Add animation keyframes
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translate(-50%, -60%);
                }
                to {
                    opacity: 1;
                    transform: translate(-50%, -50%);
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(successMessage);
        
        // Remove after animation
        setTimeout(() => {
            successMessage.remove();
            style.remove();
        }, 3000);
    }

    togglePasswordVisibility() {
        if (this.passwordInput) {
            const isPassword = this.passwordInput.type === 'password';
            this.passwordInput.type = isPassword ? 'text' : 'password';
            const toggleIcon = this.passwordToggle ? this.passwordToggle.querySelector('.toggle-icon') : null;
            if (toggleIcon) {
                toggleIcon.textContent = isPassword ? '🙈' : '👁️';
            }
        }
    }

    handleGoogleSignIn() {
        this.showSocialSignInMessage('Google');
    }

    handleAppleSignIn() {
        this.showSocialSignInMessage('Apple');
    }

    showSocialSignInMessage(provider) {
        // Create notification
        const notification = document.createElement('div');
        notification.className = 'social-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">ℹ️</div>
                <div class="notification-text">${provider} sign-in is not implemented yet</div>
            </div>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: rgba(26, 26, 26, 0.95);
            color: #00bfff;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(0, 191, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slideInRight 0.5s ease-out;
        `;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(100%);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remove after delay
        setTimeout(() => {
            notification.style.animation = 'slideInRight 0.5s ease-out reverse';
            setTimeout(() => {
                notification.remove();
                style.remove();
            }, 500);
        }, 3000);
    }

    handleDemoAccess() {
        // Show demo access message
        this.showSuccessMessage();
        
        // Navigate to main app
        setTimeout(() => {
            this.navigateToMainApp();
        }, 1500);
    }

    handleSignupLink(e) {
        e.preventDefault();
        // Show signup notification
        this.showSocialSignInMessage('Sign up feature');
    }

    handleForgotPassword(e) {
        e.preventDefault();
        // Show forgot password notification
        this.showSocialSignInMessage('Password reset');
    }

    navigateToMainApp() {
        // Send message to main process to navigate to main app
        ipcRenderer.send('navigate-to-main');
        
        // Alternative: reload with main page
        // window.location.href = 'index.html';
    }
}

// ===== INITIALIZATION =====
// Initialize the appropriate class based on the current page
document.addEventListener('DOMContentLoaded', () => {
    const currentPage = window.location.pathname;
    
    if (currentPage.includes('camera.html')) {
        const cameraManager = new CameraManager();
        console.log('Camera Manager initialized');
    } else if (currentPage.includes('overlay.html')) {
        const overlay = new SpynOverlay();
        console.log('Spyn Overlay initialized');
    } else if (currentPage.includes('signin.html')) {
        const signInPage = new SignInPage();
        console.log('Sign-in page initialized');
    } else {
        // Default to main dashboard
        const app = new Spyn();
        console.log('Spyn initialized');
    }
});

// Handle navigation from main process
ipcRenderer.on('navigate-to-signin', () => {
    window.location.href = 'signin.html';
});
