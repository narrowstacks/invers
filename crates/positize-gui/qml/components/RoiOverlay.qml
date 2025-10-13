// ROI Selection Overlay
// TODO: Implement in M2

import QtQuick 2.15

Item {
    id: root

    property rect roiRect: Qt.rect(0, 0, 0, 0)
    property bool active: false

    // Semi-transparent overlay outside ROI
    Canvas {
        id: overlay
        anchors.fill: parent
        visible: active

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)

            // Darken area outside ROI
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)"
            ctx.fillRect(0, 0, width, height)

            // Clear ROI area
            ctx.clearRect(roiRect.x, roiRect.y, roiRect.width, roiRect.height)
        }
    }

    // ROI rectangle border
    Rectangle {
        visible: active
        x: roiRect.x
        y: roiRect.y
        width: roiRect.width
        height: roiRect.height
        color: "transparent"
        border.color: "#00ff00"
        border.width: 2

        // Corner handles
        Repeater {
            model: 4
            Rectangle {
                width: 8
                height: 8
                color: "#00ff00"
                radius: 4
                // TODO: Position at corners
                // TODO: Allow dragging to resize
            }
        }
    }

    // Statistics display
    Rectangle {
        visible: active && roiRect.width > 0
        x: roiRect.x
        y: roiRect.y - height - 5
        width: 200
        height: 60
        color: "#cc000000"
        radius: 4

        // TODO: Display median RGB values
        // TODO: Display noise statistics
    }
}
