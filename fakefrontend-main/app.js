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
        this.exerciseActive = false;
        this.selectedExercise = null;
        
        this.initializeElements();
        this.bindEvents();
        this.initializeChart();
        this.setupIPC();
        this.initializeVoiceHelpers();
        
        // Start metrics checking immediately to show current data
        this.startMetricsChecking();
    }

    initializeElements() {
        // Dashboard elements
        this.startBtn = document.getElementById('startMonitoringBtn');
        this.postureReport = document.getElementById('postureReport');
        
        // Exercise elements
        this.startExerciseBtn = document.getElementById('startExerciseBtn');
        this.exerciseAnalysis = document.getElementById('exerciseAnalysis');
        this.exerciseCards = document.querySelectorAll('.exercise-card');
        this.exerciseSelectionMessage = document.getElementById('exerciseSelectionMessage');
        
        // Modal elements
        this.shortcutsBtn = document.getElementById('shortcutsBtn');
        this.shortcutsModal = document.getElementById('shortcutsModal');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        
        // Sign-out element
        this.signoutBtn = document.getElementById('signoutBtn');
        
        // Voice helper toggles
        this.voiceToggleMain = document.getElementById('voiceToggleMain');
        this.voiceToggleHero = document.getElementById('voiceToggleHero');
        
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
        
        // Add stop exercise button event
        const stopExerciseBtn = document.getElementById('stopExerciseBtn');
        if (stopExerciseBtn) {
            stopExerciseBtn.addEventListener('click', () => this.stopExerciseAnalysis());
        }
        
        // Exercise selection events
        this.exerciseCards.forEach(card => {
            card.addEventListener('click', () => this.selectExercise(card));
        });
        
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
        
        // Voice helper toggles
        if (this.voiceToggleMain) {
            this.voiceToggleMain.addEventListener('click', () => this.toggleVoiceHelpers());
        }
        if (this.voiceToggleHero) {
            this.voiceToggleHero.addEventListener('click', () => this.toggleVoiceHelpers());
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

    async switchTab(tabName) {
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
                await this.stopMonitoring();
            }
        }
    }

    setupIPC() {
        // Listen for messages from main process
        ipcRenderer.on('overlay-closed', async () => {
            await this.stopMonitoring();
        });
    }

    async toggleMonitoring() {
        if (this.isMonitoring) {
            await this.stopMonitoring();
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
        
        // Initialize dashboard variables
        this.initializeDashboardVariables();
        
        // Start timer
        this.startTimer();
        
        // Start metrics checking instead of simulation
        this.startMetricsChecking();
        
        // Start fast_demo process
        this.startFastDemo();
        
        // Notify main process to start monitoring
        ipcRenderer.invoke('start-monitoring');
        
        console.log('Spyn monitoring started');
    }

    async stopMonitoring() {
        this.isMonitoring = false;
        
        // Update UI
        if (this.startBtn) {
            this.startBtn.innerHTML = '<span class="btn-icon">▶</span>Start Spyn Monitoring';
            this.startBtn.style.background = 'linear-gradient(135deg, #00bfff, #0099cc)';
            this.startBtn.style.boxShadow = '0 8px 32px rgba(0, 191, 255, 0.3)';
        }
        
        // Show report
        await this.showPostureReport();
        
        // Stop timer
        clearInterval(this.timerInterval);
        
        // Stop metrics checking
        this.stopMetricsChecking();
        
        // Stop fast_demo process
        this.stopFastDemo();
        
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

    // New method: Start checking metrics.json every 0.1 seconds
    startMetricsChecking() {
        this.metricsInterval = setInterval(() => {
            this.checkMetricsAndUpdateStatus();
        }, 100); // Check every 0.1 seconds (100ms)
    }

    // New method: Stop metrics checking
    stopMetricsChecking() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = null;
        }
    }

    // New method: Check metrics.json and update overlay status
    async checkMetricsAndUpdateStatus() {
        try {
            console.log('Dashboard: Checking metrics...');
            const response = await fetch('http://localhost:8000/metrics');
            if (response.ok) {
                const metrics = await response.json();
                console.log('Dashboard: Fetched metrics:', metrics);
                
                // Always update dashboard variables regardless of monitoring status
                this.updateDashboardVariables(metrics);
                
                // Check the score in last_event
                if (metrics.last_event && metrics.last_event.score !== undefined) {
                    const score = metrics.last_event.score;
                    const isGoodPosture = score >= 85;
                    
                    // Update posture status if it changed
                    if (this.isGoodPosture !== isGoodPosture) {
                        this.isGoodPosture = isGoodPosture;
                        console.log(`Posture status updated: ${isGoodPosture ? 'Good' : 'Bad'} (score: ${score})`);
                    }
                    
                    // Always update overlay with current score (even if status didn't change)
                    this.updateOverlayStatus(score, metrics);
                    
                    // Update posture data for tracking
                    this.postureData.push({
                        timestamp: Date.now(),
                        isGood: isGoodPosture,
                        score: score,
                        percentage: this.calculateGoodPosturePercentage()
                    });
                } else {
                    console.log('Dashboard: No last_event or score in metrics');
                }
            } else {
                console.log('Dashboard: Failed to fetch metrics, status:', response.status);
            }
        } catch (error) {
            console.error('Error checking metrics:', error);
        }
    }

    // New method: Update overlay status based on current posture
    updateOverlayStatus(currentScore, metrics = null) {
        const statusData = {
            isGood: this.isGoodPosture,
            score: currentScore || (this.postureData.length > 0 ? this.postureData[this.postureData.length - 1].score : 0),
            metrics: metrics // Include full metrics data
        };
        console.log('Dashboard: Sending status update:', statusData);
        // Send status update to overlay window
        ipcRenderer.invoke('detection-status', statusData);
    }

    // New method: Initialize dashboard variables
    initializeDashboardVariables() {
        try {
            // Initialize with default values
            if (this.overallScore) {
                this.overallScore.textContent = '0%';
            }
            
            if (this.goodPostureTime) {
                this.goodPostureTime.textContent = '0%';
            }
            
            if (this.corrections) {
                this.corrections.textContent = '0';
            }
            
            if (this.sessionDuration) {
                this.sessionDuration.textContent = '00:00';
            }
            
            console.log('Dashboard: Variables initialized');
        } catch (error) {
            console.error('Error initializing dashboard variables:', error);
        }
    }

    // New method: Update dashboard variables in real-time
    updateDashboardVariables(metrics) {
        try {
            console.log('Dashboard: Updating variables with metrics:', metrics);
            console.log('Dashboard: Element availability:', {
                overallScore: !!this.overallScore,
                sessionDuration: !!this.sessionDuration,
                goodPostureTime: !!this.goodPostureTime,
                corrections: !!this.corrections
            });
            
            // Update overall score
            if (this.overallScore && metrics.overall_score !== undefined) {
                this.overallScore.textContent = `${metrics.overall_score}%`;
                console.log(`Dashboard: Updated overall score to ${metrics.overall_score}%`);
            } else {
                console.log('Dashboard: overallScore element not found or no overall_score in metrics');
            }
            
            // Update good posture percentage
            if (this.goodPostureTime && metrics.good_posture_pct !== undefined) {
                this.goodPostureTime.textContent = `${metrics.good_posture_pct}%`;
                console.log(`Dashboard: Updated good posture time to ${metrics.good_posture_pct}%`);
            } else {
                console.log('Dashboard: goodPostureTime element not found or no good_posture_pct in metrics');
            }
            
            // Update corrections count
            if (this.corrections && metrics.corrections !== undefined) {
                this.corrections.textContent = metrics.corrections.toString();
                console.log(`Dashboard: Updated corrections to ${metrics.corrections}`);
            } else {
                console.log('Dashboard: corrections element not found or no corrections in metrics');
            }
            
            // Update session duration using metrics data
            if (this.sessionDuration && metrics.session_duration_sec !== undefined) {
                const sessionDuration = metrics.session_duration_sec;
                const minutes = Math.floor(sessionDuration / 60);
                const seconds = sessionDuration % 60;
                this.sessionDuration.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                console.log(`Dashboard: Updated session duration to ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
            } else {
                console.log('Dashboard: sessionDuration element not found or no session_duration_sec in metrics');
            }
            
            console.log('Dashboard: Variables updated with real-time metrics');
        } catch (error) {
            console.error('Error updating dashboard variables:', error);
        }
    }

    // New method: Start fast_demo process
    startFastDemo() {
        // Send IPC message to main process to start fast_demo
        ipcRenderer.invoke('start-fast-demo');
        console.log('Fast demo start requested');
    }

    // New method: Stop fast_demo process
    stopFastDemo() {
        // Send IPC message to main process to stop fast_demo
        ipcRenderer.invoke('stop-fast-demo');
        console.log('Fast demo stop requested');
    }

    async showPostureReport() {
        try {
            // Fetch real-time metrics from the API
            const response = await fetch('http://localhost:8000/metrics');
            const metrics = await response.json();
            
            // Calculate session duration
            const sessionDuration = this.sessionStartTime ? 
                Math.floor((Date.now() - this.sessionStartTime) / 1000) : 0;
            
            const minutes = Math.floor(sessionDuration / 60);
            const seconds = sessionDuration % 60;
            
            // Use real-time values from metrics.json
            const overallScore = metrics.overall_score || 0;
            const goodPosturePct = metrics.good_posture_pct || 0;
            const corrections = metrics.corrections || 0;
            
            // Update report metrics with real-time values
            if (this.overallScore) this.overallScore.textContent = `${overallScore}%`;
            if (this.sessionDuration) this.sessionDuration.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            if (this.goodPostureTime) this.goodPostureTime.textContent = `${goodPosturePct}%`;
            if (this.corrections) this.corrections.textContent = corrections.toString();
            
            // Show report
            if (this.postureReport) {
                this.postureReport.classList.remove('hidden');
            }
            
            // Update chart
            this.updateChart();
            
            console.log('Dashboard updated with real-time metrics:', {
                overallScore,
                goodPosturePct,
                corrections,
                sessionDuration: `${minutes}:${seconds.toString().padStart(2, '0')}`
            });
            
        } catch (error) {
            console.error('Error fetching metrics for dashboard:', error);
            
            // Fallback to calculated values if API fails
            const sessionDuration = this.sessionStartTime ? 
                Math.floor((Date.now() - this.sessionStartTime) / 1000) : 0;
            
            const minutes = Math.floor(sessionDuration / 60);
            const seconds = sessionDuration % 60;
            
            const goodPosturePercentage = this.calculateGoodPosturePercentage();
            const corrections = this.postureData.filter((data, index) => 
                index > 0 && data.isGood !== this.postureData[index - 1].isGood
            ).length;
            
            // Update report metrics with fallback values
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

    selectExercise(card) {
        // Remove selected class from all cards
        this.exerciseCards.forEach(c => c.classList.remove('selected'));
        
        // Add selected class to clicked card
        card.classList.add('selected');
        
        // Get exercise name from data attribute
        this.selectedExercise = card.getAttribute('data-exercise');
        
        // Update status text
        const statusElement = card.querySelector('.exercise-status');
        if (statusElement) {
            statusElement.textContent = 'Selected';
        }
        
        // Hide selection message
        if (this.exerciseSelectionMessage) {
            this.exerciseSelectionMessage.style.display = 'none';
        }
        
        // Enable start button
        if (this.startExerciseBtn) {
            this.startExerciseBtn.disabled = false;
            this.startExerciseBtn.style.opacity = '1';
        }
        
        console.log('Exercise selected:', this.selectedExercise);
    }

    startExerciseAnalysis() {
        // Check if exercise is selected
        if (!this.selectedExercise) {
            if (this.exerciseSelectionMessage) {
                this.exerciseSelectionMessage.style.display = 'block';
            }
            return;
        }
        
        if (this.exerciseActive) {
            // Stop exercise analysis
            this.stopExerciseAnalysis();
        } else {
            // Start exercise analysis
            if (this.exerciseAnalysis) {
                this.exerciseAnalysis.classList.remove('hidden');
                
                // Update button text and state
                if (this.startExerciseBtn) {
                    this.startExerciseBtn.textContent = 'Stop Exercise Analysis';
                    this.startExerciseBtn.classList.remove('start-btn');
                    this.startExerciseBtn.classList.add('stop-btn');
                }
                
                // Initialize camera for exercise analysis
                this.initializeExerciseCamera();
                
                // Start fast_demo mode 2 (EXERCISE) instead of simulation
                this.startFastDemoExercise();
                
                // Start exercise metrics checking
                this.startExerciseMetricsChecking();
                
                // Start voice assistant for exercise feedback
                this.startExerciseVoiceAssistant();
                
                this.exerciseActive = true;
                console.log('Exercise analysis started for:', this.selectedExercise);
            }
        }
    }

    async initializeExerciseCamera() {
        const videoElement = document.getElementById('exerciseVideoElement');
        const placeholder = document.getElementById('exerciseCameraPlaceholder');
        const errorElement = document.getElementById('exerciseCameraError');

        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });

            // Set up video element
            if (videoElement) {
                videoElement.srcObject = stream;
                videoElement.style.display = 'block';
                if (placeholder) placeholder.style.display = 'none';
                if (errorElement) errorElement.style.display = 'none';
                
                console.log('Exercise camera initialized successfully');
            }
        } catch (error) {
            console.error('Error accessing camera for exercise analysis:', error);
            
            // Show error message
            if (errorElement) {
                errorElement.textContent = `Camera access failed: ${error.message}`;
                errorElement.style.display = 'block';
            }
            
            // Keep placeholder visible
            if (placeholder) {
                placeholder.innerHTML = `
                    <div class="camera-icon">❌</div>
                    <p>Camera access denied or unavailable</p>
                `;
            }
        }
    }

    stopExerciseAnalysis() {
        // Stop camera stream
        const videoElement = document.getElementById('exerciseVideoElement');
        if (videoElement && videoElement.srcObject) {
            const tracks = videoElement.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
            videoElement.style.display = 'none';
        }

        // Show placeholder again
        const placeholder = document.getElementById('exerciseCameraPlaceholder');
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.innerHTML = `
                <div class="camera-icon">📹</div>
                <p>Camera feed will appear here</p>
            `;
        }

        // Hide error message
        const errorElement = document.getElementById('exerciseCameraError');
        if (errorElement) {
            errorElement.style.display = 'none';
        }

        // Clear exercise interval
        if (this.exerciseInterval) {
            clearInterval(this.exerciseInterval);
            this.exerciseInterval = null;
        }

        // Stop exercise metrics checking
        this.stopExerciseMetricsChecking();

        // Stop voice assistant
        this.stopExerciseVoiceAssistant();

        // Stop fast_demo exercise
        this.stopFastDemoExercise();

        // Reset button
        if (this.startExerciseBtn) {
            this.startExerciseBtn.textContent = 'Start Exercise Analysis';
            this.startExerciseBtn.classList.remove('stop-btn');
            this.startExerciseBtn.classList.add('start-btn');
        }

        // Hide analysis section
        if (this.exerciseAnalysis) {
            this.exerciseAnalysis.classList.add('hidden');
        }

        // Reset exercise selection
        this.exerciseCards.forEach(card => {
            card.classList.remove('selected');
            const statusElement = card.querySelector('.exercise-status');
            if (statusElement) {
                statusElement.textContent = 'Not Selected';
            }
        });
        this.selectedExercise = null;

        this.exerciseActive = false;
        console.log('Exercise analysis stopped');
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

    // New method: Start fast_demo mode 2 (EXERCISE)
    startFastDemoExercise() {
        console.log('Starting fast_demo exercise mode...');
        // Send IPC message to main process to start fast_demo with exercise mode
        ipcRenderer.invoke('start-fast-demo-exercise');
    }

    // New method: Stop fast_demo exercise
    stopFastDemoExercise() {
        console.log('Stopping fast_demo exercise mode...');
        // Send IPC message to main process to stop fast_demo
        ipcRenderer.invoke('stop-fast-demo');
    }

    // New method: Start exercise metrics checking
    startExerciseMetricsChecking() {
        this.exerciseMetricsInterval = setInterval(() => {
            this.checkExerciseMetricsAndUpdate();
        }, 100); // Check every 0.1 seconds (100ms) for more responsive updates
    }

    // New method: Stop exercise metrics checking
    stopExerciseMetricsChecking() {
        if (this.exerciseMetricsInterval) {
            clearInterval(this.exerciseMetricsInterval);
            this.exerciseMetricsInterval = null;
        }
    }

    // New method: Check exercise metrics and update display
    async checkExerciseMetricsAndUpdate() {
        try {
            console.log('Exercise: Checking metrics...');
            const response = await fetch('http://localhost:8000/metrics');
            if (response.ok) {
                const metrics = await response.json();
                console.log('Exercise: Fetched metrics:', metrics);
                this.updateExerciseVariables(metrics);
            } else {
                console.log('Exercise: Failed to fetch metrics, status:', response.status);
            }
        } catch (error) {
            console.error('Exercise: Error checking metrics:', error);
        }
    }

    // New method: Update exercise variables from metrics
    updateExerciseVariables(metrics) {
        try {
            console.log('Exercise: Updating variables with metrics:', metrics);
            
            // Update form score (use last_event score - current exercise form quality)
            const formScore = document.getElementById('formScore');
            if (formScore && metrics.last_event && metrics.last_event.score !== undefined) {
                formScore.textContent = `${metrics.last_event.score}%`;
                console.log(`Exercise: Updated form score to ${metrics.last_event.score}%`);
            }
            
            // Update reps count (use overall_score as a proxy for exercise progress)
            const repsCount = document.getElementById('repsCount');
            if (repsCount && metrics.overall_score !== undefined) {
                // Use overall_score as a rough indicator of exercise progress
                repsCount.textContent = Math.floor(metrics.overall_score / 10).toString();
                console.log(`Exercise: Updated reps count to ${Math.floor(metrics.overall_score / 10)}`);
            }
            
            // Update corrections made (use actual corrections count)
            const correctionsMade = document.getElementById('correctionsMade');
            if (correctionsMade && metrics.corrections !== undefined) {
                correctionsMade.textContent = metrics.corrections.toString();
                console.log(`Exercise: Updated corrections to ${metrics.corrections}`);
            }
            
            // Update session time (use session_duration_sec)
            const timeElapsed = document.getElementById('timeElapsed');
            if (timeElapsed && metrics.session_duration_sec !== undefined) {
                const sessionDuration = metrics.session_duration_sec;
                const minutes = Math.floor(sessionDuration / 60);
                const seconds = sessionDuration % 60;
                timeElapsed.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                console.log(`Exercise: Updated session time to ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
            }
            
            // Store current feedback for voice assistant
            if (metrics.last_event && metrics.last_event.feedback) {
                this.currentExerciseFeedback = metrics.last_event.feedback;
            }
            
            // Store current form score for voice feedback
            if (metrics.last_event && metrics.last_event.score !== undefined) {
                this.currentFormScore = metrics.last_event.score;
            }
            
            console.log('Exercise: Variables updated successfully');
        } catch (error) {
            console.error('Exercise: Error updating variables:', error);
        }
    }

    // New method: Start voice assistant for exercise feedback
    startExerciseVoiceAssistant() {
        this.exerciseVoiceInterval = setInterval(() => {
            this.speakExerciseFeedback();
        }, 2000); // Speak every 2 seconds for more responsive feedback
    }

    // New method: Stop voice assistant
    stopExerciseVoiceAssistant() {
        if (this.exerciseVoiceInterval) {
            clearInterval(this.exerciseVoiceInterval);
            this.exerciseVoiceInterval = null;
        }
    }

    // New method: Speak exercise feedback
    async speakExerciseFeedback() {
        if (!this.currentExerciseFeedback || this.currentExerciseFeedback.length === 0) {
            return;
        }

        try {
            // Get the first feedback item (most important)
            const feedback = this.currentExerciseFeedback[0];
            if (feedback && !feedback.startsWith('⚠️')) {
                console.log('Exercise: Speaking feedback:', feedback);
                
                // Format feedback for voice - make it more natural
                let voiceText = feedback;
                if (this.currentFormScore !== undefined) {
                    if (this.currentFormScore >= 85) {
                        voiceText = `Excellent! ${feedback} Your score is ${this.currentFormScore} percent.`;
                    } else if (this.currentFormScore >= 70) {
                        voiceText = `${feedback} Current score: ${this.currentFormScore} percent. Keep improving!`;
                    } else {
                        voiceText = `${feedback} Score: ${this.currentFormScore} percent. Focus on form.`;
                    }
                }
                
                // Send to backend for text-to-speech
                const response = await fetch('http://localhost:8000/speak', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: voiceText })
                });

                if (response.ok) {
                    // Create audio element and play
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    audio.play().catch(error => {
                        console.error('Exercise: Error playing audio:', error);
                    });
                } else {
                    console.error('Exercise: Failed to get speech, status:', response.status);
                }
            }
        } catch (error) {
            console.error('Exercise: Error speaking feedback:', error);
        }
    }

    initializeKeyboardShortcuts() {
        // Add keyboard event listener to document
        document.addEventListener('keydown', async (e) => {
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
                    case 'v':
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.toggleVoiceHelpers();
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
                        await this.stopMonitoring();
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

    initializeVoiceHelpers() {
        // Initialize voice helper state from localStorage
        this.voiceHelpersEnabled = localStorage.getItem('voiceHelpersEnabled') === 'true';
        this.updateVoiceToggleState();
    }
    
    updateVoiceToggleState() {
        // Update main dashboard voice toggle
        if (this.voiceToggleMain) {
            const voiceIcon = this.voiceToggleMain.querySelector('.voice-icon');
            const voiceText = this.voiceToggleMain.querySelector('.voice-text');
            
            if (this.voiceHelpersEnabled) {
                this.voiceToggleMain.classList.add('active');
                if (voiceIcon) voiceIcon.textContent = '🔊';
                if (voiceText) voiceText.textContent = 'ON';
                this.voiceToggleMain.title = 'Voice Helpers ON - Click to disable';
            } else {
                this.voiceToggleMain.classList.remove('active');
                if (voiceIcon) voiceIcon.textContent = '🔇';
                if (voiceText) voiceText.textContent = 'OFF';
                this.voiceToggleMain.title = 'Voice Helpers OFF - Click to enable';
            }
        }
        
        // Update hero section voice toggle
        if (this.voiceToggleHero) {
            const voiceIcon = this.voiceToggleHero.querySelector('.voice-icon');
            const voiceText = this.voiceToggleHero.querySelector('.voice-text');
            
            if (this.voiceHelpersEnabled) {
                this.voiceToggleHero.classList.add('active');
                if (voiceIcon) voiceIcon.textContent = '🔊';
                if (voiceText) voiceText.textContent = 'Voice ON';
                this.voiceToggleHero.title = 'Voice Helpers ON - Click to disable (Ctrl+Shift+V)';
            } else {
                this.voiceToggleHero.classList.remove('active');
                if (voiceIcon) voiceIcon.textContent = '🔇';
                if (voiceText) voiceText.textContent = 'Voice OFF';
                this.voiceToggleHero.title = 'Voice Helpers OFF - Click to enable (Ctrl+Shift+V)';
            }
        }
    }
    
    toggleVoiceHelpers() {
        this.voiceHelpersEnabled = !this.voiceHelpersEnabled;
        localStorage.setItem('voiceHelpersEnabled', this.voiceHelpersEnabled.toString());
        this.updateVoiceToggleState();
        
        console.log('Voice helpers:', this.voiceHelpersEnabled ? 'ENABLED' : 'DISABLED');
        
        // Show notification
        this.showVoiceToggleNotification();
    }
    
    showVoiceToggleNotification() {
        // Create notification
        const notification = document.createElement('div');
        notification.className = 'voice-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">${this.voiceHelpersEnabled ? '🔊' : '🔇'}</div>
                <div class="notification-text">Voice helpers ${this.voiceHelpersEnabled ? 'enabled' : 'disabled'}</div>
            </div>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: rgba(26, 26, 26, 0.95);
            color: ${this.voiceHelpersEnabled ? '#00ff88' : '#888'};
            padding: 1rem 1.5rem;
            border-radius: 8px;
            border: 1px solid ${this.voiceHelpersEnabled ? 'rgba(0, 255, 136, 0.3)' : 'rgba(136, 136, 136, 0.3)'};
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

    async handleSignOut() {
        // Stop monitoring if active
        if (this.isMonitoring) {
            await this.stopMonitoring();
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

        // Listen for detection status updates from main process
        ipcRenderer.on('detection-status', (event, status) => {
            this.handleDetectionStatus(status);
        });
        
        // Listen for percentage updates from main process
        ipcRenderer.on('update-percentage', (event, data) => {
            console.log('Overlay: Received percentage update:', data);
            this.updatePosturePercentage(data.score);
        });
    }

    // New method: Handle detection status updates
    handleDetectionStatus(status) {
        console.log('Overlay: Received detection status:', status);
        if (status && typeof status.isGood !== 'undefined') {
            this.isGoodPosture = status.isGood;
            this.updatePostureStatus();
            
            // Use the actual score from metrics.json instead of calculated percentage
            if (status.score !== undefined) {
                this.updatePosturePercentage(status.score);
            } else {
                this.updatePosturePercentage();
            }
            
            // Update timer with metrics data if available
            if (status.metrics) {
                this.updateTimer(status.metrics);
            }
            
            console.log(`Overlay: Posture status updated to ${this.isGoodPosture ? 'Good' : 'Bad'} (score: ${status.score || 'N/A'})`);
        }
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
    
    updateTimer(metrics = null) {
        let timeString = '00:00';
        
        // Use metrics data if available, otherwise fall back to local calculation
        if (metrics && metrics.session_duration_sec !== undefined) {
            const sessionDuration = metrics.session_duration_sec;
            const minutes = Math.floor(sessionDuration / 60);
            const seconds = sessionDuration % 60;
            timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        } else if (this.sessionStartTime) {
            const elapsed = Date.now() - this.sessionStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
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

    updatePosturePercentage(score) {
        // Use the actual score from metrics.json if provided, otherwise calculate percentage
        const percentage = score !== undefined ? score : this.calculateGoodPosturePercentage();
        if (this.posturePercentage) {
            this.posturePercentage.textContent = `${percentage}%`;
            console.log(`Overlay: Percentage updated to ${percentage}%`);
        } else {
            console.log('Overlay: posturePercentage element not found');
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
