// Image Viewer Component
// TODO: Implement in M2

import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: root

    property real zoomLevel: 1.0
    property point panOffset: Qt.point(0, 0)
    property bool showHistogram: true
    property bool showPixelProbe: false

    // Image display
    Image {
        id: imageDisplay
        anchors.centerIn: parent
        fillMode: Image.PreserveAspectFit
        smooth: true

        // TODO: Load preview image from Rust backend
        // TODO: Apply zoom and pan transformations
    }

    // Histogram overlay
    Rectangle {
        id: histogramOverlay
        visible: showHistogram
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: 20
        width: 256
        height: 100
        color: "#80000000"

        // TODO: Display histogram from backend
    }

    // Pixel probe overlay
    Rectangle {
        id: pixelProbe
        visible: showPixelProbe
        width: 200
        height: 100
        color: "#cc000000"

        // TODO: Follow mouse cursor
        // TODO: Display RGB values
    }

    // Mouse interaction
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        onWheel: {
            // TODO: Zoom on wheel
        }

        onPressed: {
            // TODO: Start pan or ROI selection
        }

        onPositionChanged: {
            // TODO: Pan or update ROI
        }
    }
}
