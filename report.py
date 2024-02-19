from logging import Logger
import requests
import os



class WorldReporter:
    def __init__(self, server_url: str, folder: str, logger: Logger) -> None:
        self._server_url = server_url
        self._logger = logger
        self._folder = folder

    def _create_report(self, camera_ip: str, start_tracking: str, stop_tracking: str, reports: list) -> dict:
        report = {
            "camera": camera_ip,
            "algorithm": os.environ.get('algorithm_name'),
            "start_tracking": start_tracking,
            "stop_tracking": stop_tracking,
            "photos": [{"date": start_tracking}],
            "violation_found": False,
            "extra": [
                {
                    "zoneId": 1,
                    "zoneName": 'main',
                    "items": reports
                }
            ]
        }
        return report

    def create_item_report(self, photo_path: str, itemId: int, count: int) -> dict:
        report = {
          'itemId': itemId,
          'count': count,
          'image_item': photo_path,
          'low_stock_level': False,
          'zoneId': 1,
          'zoneName': 'main'
        }
        return report

    def send_report(self, start_tracking: str, stop_tracking: str, camera_ip: str, final_report: list):
        try:
            report = self._create_report(camera_ip, start_tracking, stop_tracking, final_report)
            requests.post(url=os.environ.get("link_reports"), json=report)
            self._logger.info("<<<<<<<<<<<<<<<<<SEND REPORT!!!!!!!>>>>>>>>>>>>>>\n" + str(report))
        except Exception as exc:
            self._logger.critical("Error while sending report occurred: {}".format(exc))
