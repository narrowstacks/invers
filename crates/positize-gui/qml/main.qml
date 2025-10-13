// Main QML Window
// TODO: Implement in M2 with cxx-qt

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: root
    visible: true
    width: 1280
    height: 800
    title: "Positize - Film Negative Converter"

    // Main layout
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // Left sidebar - Import and Preset Manager
        Rectangle {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            color: "#2b2b2b"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                // Import Panel
                GroupBox {
                    Layout.fillWidth: true
                    title: "Import"

                    // TODO: File picker, drag-and-drop area
                }

                // Preset Manager
                GroupBox {
                    Layout.fillWidth: true
                    title: "Presets"

                    // TODO: Film preset selector
                    // TODO: Scan profile selector
                }

                // ROI Tool
                GroupBox {
                    Layout.fillWidth: true
                    title: "Base Estimation"

                    // TODO: ROI selection controls
                    // TODO: Numeric readout
                }

                Item { Layout.fillHeight: true }
            }
        }

        // Center - Image Viewer
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#1e1e1e"

            // TODO: Image viewer with zoom/pan
            // TODO: ROI overlay
            // TODO: Histogram overlay
        }

        // Right sidebar - Export and Batch Queue
        Rectangle {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            color: "#2b2b2b"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                // Export Settings
                GroupBox {
                    Layout.fillWidth: true
                    title: "Export"

                    // TODO: Format selector
                    // TODO: Colorspace selector
                    // TODO: Output directory
                }

                // Batch Queue
                GroupBox {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    title: "Batch Queue"

                    // TODO: List of batch items
                    // TODO: Progress bars
                    // TODO: Start/stop/clear buttons
                }
            }
        }
    }
}
