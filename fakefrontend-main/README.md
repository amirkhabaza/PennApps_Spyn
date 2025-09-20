# PostureMAX - Posture Correction Desktop App

A modern, sleek desktop application designed to help users maintain good posture throughout their workday. Built with a black and bright blue Web 3.0 aesthetic, PostureMAX provides real-time posture monitoring with an intuitive overlay interface.

## Features

### Dashboard
- **Modern UI**: Clean, minimalist design with gradient backgrounds and glassmorphism effects
- **Start Monitoring Button**: Large, prominent button to begin posture tracking
- **Posture Report**: Comprehensive analytics displayed after monitoring sessions including:
  - Overall posture score percentage
  - Session duration
  - Good posture time percentage
  - Number of posture corrections
  - Visual chart showing posture quality over time

### Overlay Window
- **Draggable Interface**: Move the overlay anywhere on your screen when posture is good
- **Real-time Timer**: Shows current session length
- **Posture Status Indicator**: Visual feedback showing current posture state (good/bad)
- **Mode Selector**: Dropdown to switch between different display modes:
  - Live Camera Feed (placeholder for MediaPipe integration)
  - Current Posture Status
  - Posture Percentage Display
- **Transparency Controls**: Three levels of visibility:
  - Hidden: Completely transparent
  - Semi: 30% opacity
  - Full: 100% opacity

### Smart Behavior
- **Good Posture**: Overlay remains draggable and positioned where user placed it
- **Bad Posture**: Overlay automatically centers on screen, becomes solid, and pulses until posture is corrected
- **Visual Feedback**: Color-coded status indicators and smooth animations

## Getting Started

1. **Open the Application**: Simply open `index.html` in your web browser
2. **Start Monitoring**: Click the "Start PostureMAX Monitoring" button
3. **Configure Overlay**: Use the dropdown and transparency controls to customize your experience
4. **Monitor Your Posture**: The overlay will provide real-time feedback
5. **View Report**: Stop monitoring to see detailed analytics and charts

## Technical Implementation

### Architecture
- **Frontend**: Pure HTML5, CSS3, and JavaScript (ES6+)
- **Styling**: Modern CSS with gradients, backdrop filters, and smooth animations
- **Responsive**: Works on different screen sizes with mobile-friendly adjustments
- **Modular**: Clean, object-oriented JavaScript architecture

### Key Components
- `PostureMAX` class: Main application controller
- Dashboard management
- Overlay behavior and drag functionality
- Posture simulation (placeholder for AI integration)
- Chart rendering and data visualization
- Real-time timer and status updates

### Future Integration Points
- **MediaPipe Integration**: Camera feed and posture detection
- **Machine Learning**: Real posture analysis algorithms
- **Desktop App Wrapper**: Electron or similar for native desktop experience
- **Data Persistence**: Session history and long-term analytics

## Design Philosophy

PostureMAX follows modern UI/UX principles:

- **Minimalist**: Clean interface that doesn't distract from work
- **Accessible**: High contrast colors and clear visual feedback
- **Responsive**: Adapts to different screen sizes and user preferences
- **Modern**: Web 3.0 aesthetic with glassmorphism and gradient effects
- **Intuitive**: Simple controls that are easy to understand and use

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Development Notes

The current implementation includes:
- ✅ Complete UI/UX framework
- ✅ Dashboard with monitoring controls
- ✅ Draggable overlay window
- ✅ Posture simulation (for testing)
- ✅ Real-time timer and status updates
- ✅ Transparency controls
- ✅ Visual analytics and charts
- ✅ Smart overlay behavior (drag vs center-lock)
- ✅ Modern styling and animations

**Next Steps**: Integrate actual posture detection using computer vision libraries like MediaPipe or OpenCV.

## File Structure

```
PostureMAX/
├── index.html          # Main application file
├── styles.css          # All styling and animations
├── script.js           # Application logic and behavior
└── README.md          # This documentation
```

## Usage Tips

1. **Positioning**: Drag the overlay to your preferred location when posture is good
2. **Transparency**: Use semi-transparent mode if the overlay is too distracting
3. **Monitoring**: Let the app run in the background while you work
4. **Reports**: Check your posture report regularly to track improvement
5. **Customization**: Adjust overlay size and position based on your workflow

---

*Built with ❤️ for better posture and healthier work habits.*
