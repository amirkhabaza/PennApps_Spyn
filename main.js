const { app, BrowserWindow, ipcMain, screen, globalShortcut } = require('electron');
const path = require('path');

class SpynApp {
    constructor() {
        this.mainWindow = null;
        this.overlayWindow = null;
        this.cameraWindow = null;
        this.isMonitoring = false;
        this.isOverlayVisible = false;
        this.isCameraVisible = false;
        this.topLevelInterval = null;
    }

    createMainWindow() {
        // Get primary display info
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workAreaSize;

        this.mainWindow = new BrowserWindow({
            width: 1200,
            height: 800,
            minWidth: 800,
            minHeight: 600,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true
            },
            titleBarStyle: 'hiddenInset',
            title: 'Spyn',
            icon: path.join(__dirname, 'assets/icon.png'), // Optional: add app icon
            show: false, // Don't show until ready
            backgroundColor: '#0a0a0a'
        });

        // Load the sign-in page first
        this.mainWindow.loadFile('signin.html');

        // Show window when ready
        this.mainWindow.once('ready-to-show', () => {
            this.mainWindow.show();
            
            // Focus on the window
            if (process.platform === 'darwin') {
                app.dock.show();
            }
        });

        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
            if (this.overlayWindow) {
                this.overlayWindow.close();
            }
        });

        // Handle minimize to tray (optional)
        this.mainWindow.on('minimize', (event) => {
            if (process.platform === 'win32') {
                event.preventDefault();
                this.mainWindow.hide();
            }
        });

        // Open DevTools in development
        if (process.argv.includes('--dev')) {
            this.mainWindow.webContents.openDevTools();
        }
    }

    createOverlayWindow() {
        if (this.overlayWindow) {
            this.overlayWindow.focus();
            return;
        }

        // Get screen dimensions
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workAreaSize;

        // Horizontal bar layout
        this.overlayWindow = new BrowserWindow({
            width: 800,
            height: 80,
            minWidth: 700,
            minHeight: 80,
            maxWidth: 1000,
            maxHeight: 80,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false
            },
            frame: false, // Frameless window
            resizable: true,
            alwaysOnTop: true, // Always on top
            skipTaskbar: true, // Don't show in taskbar
            transparent: true, // Enable transparency
            show: false,
            backgroundColor: '#00000000', // Transparent background
            x: Math.round((width - 800) / 2), // Center horizontally
            y: 50, // Position near top
            opacity: 1.0,
            // Additional properties for true persistence
            fullscreenable: false,
            maximizable: false,
            minimizable: false,
            closable: true,
            focusable: false, // Prevent focus stealing
            acceptFirstMouse: true, // Allow clicking through
            // macOS specific properties to prevent desktop switching
            visibleOnAllWorkspaces: true, // Show on all desktop spaces
            fullscreenable: false, // Prevent fullscreen mode
            // Additional properties to prevent focus stealing
            showInactive: true, // Don't activate when shown
            disableAutoHideCursor: true, // Don't hide cursor
            // Prevent window from becoming active
            alwaysOnTop: true,
            skipTaskbar: true
        });

        // Load overlay HTML
        this.overlayWindow.loadFile('overlay.html');

        // Show when ready
        this.overlayWindow.once('ready-to-show', () => {
            this.overlayWindow.show();
            this.isOverlayVisible = true;
            
            // Ensure it stays on all workspaces and is visible during fullscreen
            if (process.platform === 'darwin') {
                this.overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
                // Re-enforce highest window level periodically
                this.enforceTopLevel();
            }
            
            // Ensure always on top is properly set
            this.ensureAlwaysOnTop();
        });

        // Prevent focus stealing when overlay is clicked
        this.overlayWindow.on('focus', () => {
            // Immediately blur the overlay window to prevent focus stealing
            this.overlayWindow.blur();
        });

        // Prevent activation when clicked
        this.overlayWindow.on('activate', () => {
            // Don't allow the overlay to become active
            this.overlayWindow.blur();
        });

        // Handle overlay closed
        this.overlayWindow.on('closed', () => {
            this.overlayWindow = null;
            this.isOverlayVisible = false;
            
            // Clear top level enforcement interval
            if (this.topLevelInterval) {
                clearInterval(this.topLevelInterval);
                this.topLevelInterval = null;
            }
        });

        // Prevent window from being moved to different desktop spaces
        this.overlayWindow.on('move', () => {
            if (process.platform === 'darwin') {
                // Re-enforce visibility on all workspaces when moved
                this.overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
            }
        });

        // Make overlay draggable and set highest window level
        this.overlayWindow.setMovable(true);
        
        // Set window level for maximum visibility above fullscreen apps
        if (process.platform === 'darwin') {
            // On macOS, use the highest possible level to appear above fullscreen
            this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
            // Ensure it appears on all workspaces and during fullscreen
            this.overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
        } else if (process.platform === 'win32') {
            // On Windows, use screen-saver level
            this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
        } else {
            // On Linux, use screen-saver level
            this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
        }

        // Store initial state
        this.overlayState = {
            transparency: 'visible'
        };
    }

    createCameraWindow() {
        if (this.cameraWindow) {
            this.cameraWindow.focus();
            return;
        }

        // Get screen dimensions
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workAreaSize;

        // Normal camera window
        const windowConfig = {
            width: 320,
            height: 240,
            minWidth: 280,
            minHeight: 200,
            maxWidth: 480,
            maxHeight: 360,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false
            },
            frame: true,
            resizable: true,
            alwaysOnTop: true,
            skipTaskbar: false,
            transparent: false,
            show: false,
            backgroundColor: '#000000',
            x: width - 340,
            y: 20,
            opacity: 1.0,
            title: 'Spyn Camera'
        };

        this.cameraWindow = new BrowserWindow(windowConfig);

        // Load camera HTML
        this.cameraWindow.loadFile('camera.html');

        // Keep camera window hidden initially but mark as active
        this.cameraWindow.once('ready-to-show', () => {
            // Don't show the camera window initially - it will be shown when camera mode is selected
            this.isCameraVisible = false;
            console.log('Camera window created and ready (hidden)');
        });

        // Prevent camera window from closing when monitoring is active
        this.cameraWindow.on('close', (event) => {
            if (this.isMonitoring) {
                // Prevent the window from actually closing
                event.preventDefault();
                // Just hide it instead
                this.cameraWindow.hide();
                this.isCameraVisible = false;
                console.log('Camera window hidden instead of closed (monitoring active)');
            }
        });

        // Handle camera window closed (only when actually closed)
        this.cameraWindow.on('closed', () => {
            this.cameraWindow = null;
            this.isCameraVisible = false;
        });

        // Make camera window draggable
        this.cameraWindow.setMovable(true);
        this.cameraWindow.setAlwaysOnTop(true, 'screen-saver');
    }

    startMonitoring() {
        this.isMonitoring = true;
        this.createOverlayWindow();
        
        // Create camera window but keep it hidden initially
        this.createCameraWindow();
        
        // Send start signal to overlay
        if (this.overlayWindow) {
            this.overlayWindow.webContents.send('start-monitoring');
        }
    }

    stopMonitoring() {
        this.isMonitoring = false;
        
        // Clear top level enforcement interval
        if (this.topLevelInterval) {
            clearInterval(this.topLevelInterval);
            this.topLevelInterval = null;
        }
        
        // Close camera window when monitoring stops
        if (this.cameraWindow && !this.cameraWindow.isDestroyed()) {
            this.cameraWindow.close();
            this.cameraWindow = null;
            this.isCameraVisible = false;
        }
        
        if (this.overlayWindow) {
            this.overlayWindow.webContents.send('stop-monitoring');
            // Close overlay after a short delay to show final stats
            setTimeout(() => {
                if (this.overlayWindow) {
                    this.overlayWindow.close();
                }
            }, 2000);
        }
    }

    toggleOverlay() {
        if (this.isOverlayVisible) {
            if (this.overlayWindow) {
                this.overlayWindow.close();
            }
        } else {
            this.createOverlayWindow();
            // Ensure always on top is set when creating overlay
            setTimeout(() => {
                this.ensureAlwaysOnTop();
            }, 100);
        }
    }

    showCamera() {
        if (this.cameraWindow && !this.cameraWindow.isDestroyed()) {
            // Camera window exists, show it in corner mode
            this.showCameraInCorner();
            this.isCameraVisible = true;
        } else {
            // Create new camera window and show it
            this.createCameraWindow();
            // Show it immediately after creation
            setTimeout(() => {
                if (this.cameraWindow && !this.cameraWindow.isDestroyed()) {
                    this.showCameraInCorner();
                    this.isCameraVisible = true;
                }
            }, 500);
        }
    }

    showCameraInCorner() {
        if (this.cameraWindow && !this.cameraWindow.isDestroyed()) {
            // Resize to corner window and position it
            const primaryDisplay = screen.getPrimaryDisplay();
            const { width, height } = primaryDisplay.workAreaSize;
            
            this.cameraWindow.setSize(320, 240);
            this.cameraWindow.setPosition(width - 340, 20);
            this.cameraWindow.setResizable(true);
            this.cameraWindow.setFullScreen(false);
            this.cameraWindow.setAlwaysOnTop(true, 'normal');
            // Note: setFrame is not available in all Electron versions, removing it
            
            this.cameraWindow.show();
            this.cameraWindow.focus();
            console.log('Camera window shown in corner');
        }
    }


    hideCamera() {
        if (this.cameraWindow && !this.cameraWindow.isDestroyed()) {
            this.cameraWindow.hide();
            this.isCameraVisible = false;
        }
    }

    centerOverlay() {
        if (this.overlayWindow) {
            const primaryDisplay = screen.getPrimaryDisplay();
            const { width, height } = primaryDisplay.workAreaSize;
            
            const overlayBounds = this.overlayWindow.getBounds();
            const x = Math.round((width - overlayBounds.width) / 2);
            const y = Math.round((height - overlayBounds.height) / 2);
            
            this.overlayWindow.setPosition(x, y);
        }
    }

    enforceTopLevel() {
        if (this.overlayWindow) {
            // Continuously enforce the highest window level
            const enforceLevels = () => {
                if (this.overlayWindow && this.isOverlayVisible) {
                    // Use screen-saver level for maximum visibility above fullscreen apps
                    this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
                    
                    // Platform-specific enforcement
                    if (process.platform === 'darwin') {
                        // Ensure it appears on all workspaces and during fullscreen
                        this.overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
                    }
                    
                    // Ensure focus prevention
                    this.overlayWindow.setFocusable(false);
                    this.overlayWindow.setSkipTaskbar(true);
                }
            };
            
            // Enforce immediately
            enforceLevels();
            
            // Set up periodic enforcement every 2 seconds to maintain top level
            this.topLevelInterval = setInterval(enforceLevels, 2000);
        }
    }

    ensureAlwaysOnTop() {
        if (this.overlayWindow) {
            // Set always on top with the highest possible level
            this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
            
            // Platform-specific optimizations
            if (process.platform === 'darwin') {
                // On macOS, ensure visibility on all workspaces including fullscreen
                this.overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
                // Set the highest possible window level
                this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
            } else if (process.platform === 'win32') {
                // On Windows, use screen-saver level for maximum visibility
                this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
            } else {
                // On Linux, use screen-saver level
                this.overlayWindow.setAlwaysOnTop(true, 'screen-saver');
            }
            
            // Additional properties for maximum visibility
            this.overlayWindow.setSkipTaskbar(true);
            this.overlayWindow.setFocusable(false);
            
            console.log('Overlay window configured for always on top');
        }
    }




    setOverlayTransparency(level) {
        if (this.overlayWindow) {
            this.overlayState.transparency = level;
            
            switch (level) {
                case 'transparent':
                    this.overlayWindow.setOpacity(0.3);
                    this.overlayWindow.webContents.send('transparency-changed', 'transparent');
                    break;
                case 'visible':
                    this.overlayWindow.setOpacity(1.0);
                    this.overlayWindow.webContents.send('transparency-changed', 'visible');
                    break;
            }
        }
    }

    setupGlobalShortcuts() {
        // Register global shortcuts
        globalShortcut.register('CommandOrControl+Shift+P', () => {
            this.toggleOverlay();
        });

        globalShortcut.register('CommandOrControl+Shift+M', () => {
            if (this.isMonitoring) {
                this.stopMonitoring();
            } else {
                this.startMonitoring();
            }
        });

        // Transparency shortcuts
        globalShortcut.register('CommandOrControl+Shift+T', () => {
            this.setOverlayTransparency('transparent');
        });

        globalShortcut.register('CommandOrControl+Shift+F', () => {
            this.setOverlayTransparency('visible');
        });

        // Camera shortcuts
        globalShortcut.register('CommandOrControl+Shift+C', () => {
            if (this.isCameraVisible) {
                this.hideCamera();
            } else {
                this.showCamera();
            }
        });

        // Overlay positioning shortcuts
        globalShortcut.register('CommandOrControl+Shift+O', () => {
            this.centerOverlay();
        });

        // Quick start/stop shortcuts (single key combinations)
        globalShortcut.register('F9', () => {
            if (this.isMonitoring) {
                this.stopMonitoring();
            } else {
                this.startMonitoring();
            }
        });

        globalShortcut.register('F10', () => {
            this.toggleOverlay();
        });

        globalShortcut.register('F11', () => {
            if (this.isCameraVisible) {
                this.hideCamera();
            } else {
                this.showCamera();
            }
        });

        // Function key shortcuts for transparency
        globalShortcut.register('F1', () => {
            this.setOverlayTransparency('visible');
        });

        globalShortcut.register('F2', () => {
            this.setOverlayTransparency('transparent');
        });

        console.log('Global shortcuts registered successfully');
    }

    setupIPC() {
        // Handle IPC messages from renderer process
        ipcMain.handle('start-monitoring', () => {
            this.startMonitoring();
            return { success: true };
        });

        ipcMain.handle('stop-monitoring', () => {
            this.stopMonitoring();
            return { success: true };
        });

        ipcMain.handle('toggle-overlay', () => {
            this.toggleOverlay();
            return { success: true };
        });

        ipcMain.handle('center-overlay', () => {
            this.centerOverlay();
            return { success: true };
        });



        ipcMain.handle('set-transparency', (event, level) => {
            this.setOverlayTransparency(level);
            return { success: true };
        });

        ipcMain.handle('close-overlay', () => {
            this.stopMonitoring();
            return { success: true };
        });

        ipcMain.handle('show-camera', () => {
            this.showCamera();
            return { success: true };
        });

        ipcMain.handle('hide-camera', () => {
            this.hideCamera();
            return { success: true };
        });


        ipcMain.handle('detection-status', (event, status) => {
            // Forward detection status to overlay window
            if (this.overlayWindow && !this.overlayWindow.isDestroyed()) {
                this.overlayWindow.webContents.send('detection-status', status);
            }
            return { success: true };
        });

        ipcMain.handle('get-app-version', () => {
            return app.getVersion();
        });

        // Navigation handlers
        ipcMain.on('navigate-to-main', () => {
            if (this.mainWindow) {
                this.mainWindow.loadFile('index.html');
            }
        });

        ipcMain.on('navigate-to-signin', () => {
            if (this.mainWindow) {
                this.mainWindow.loadFile('signin.html');
            }
        });
    }
}

// Create app instance
const spynApp = new SpynApp();

// App event handlers
app.whenReady().then(() => {
    spynApp.createMainWindow();
    spynApp.setupIPC();
    spynApp.setupGlobalShortcuts();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            spynApp.createMainWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('will-quit', () => {
    // Unregister all global shortcuts
    globalShortcut.unregisterAll();
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
    });
});

// Export for potential use in renderer
if (typeof module !== 'undefined' && module.exports) {
    module.exports = spynApp;
}
