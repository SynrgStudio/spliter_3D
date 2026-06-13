from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyqt6")
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = os.environ.get("SPLIT3R_QT_QPA_PLATFORM", "xcb")
else:
    os.environ.setdefault("QT_OPENGL", "desktop")

from PyQt6.QtWidgets import QApplication

from split3r_rewrite.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
