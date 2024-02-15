from logging import Logger
import requests
import os


class WorldReporter:
    def __init__(self, server_url: str, folder: str, logger: Logger) -> None:
        self._server_url = server_url
        self._logger = logger
        self._folder = folder

    def _create_report(self, photo_path: str, start_tracking: str, stop_tracking: str) -> dict:
        report = {
            "algorithm": os.environ.get('algorithm_name'),
            "camera": self._folder.split('/')[1],
            "start_tracking": start_tracking,
            "stop_tracking": stop_tracking,
            "photos": [{"image": photo_path, "date": start_tracking}],
            "violation_found": True
        }
        return report

    def send_report(self, photo_path: str, start_tracking: str, stop_tracking: str):
        try:
            report = self._create_report(photo_path, start_tracking, stop_tracking)
            requests.post(url=os.environ.get("link_reports"), json=report)
            self._logger.info("<<<<<<<<<<<<<<<<<SEND REPORT!!!!!!!>>>>>>>>>>>>>>\n" + str(report))
        except Exception as exc:
            self._logger.error("Error while sending report occurred: {}".format(exc))
            print(str(report))