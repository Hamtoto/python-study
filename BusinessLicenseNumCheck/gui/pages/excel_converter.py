import os, sys
import requests
from dotenv import load_dotenv
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QFileDialog
from components.buttons import create_styled_button
from components.text_field import StyledTextField
from utils.config import load_settings, save_settings
import pandas as pd


if getattr(sys, 'frozen', False):
    # 실행 파일로 빌드된 경우, .env 파일은 sys._MEIPASS에 복사되어 있을 가능성이 있습니다.
    dotenv_path = os.path.join(sys._MEIPASS, ".env")
    load_dotenv(dotenv_path)
else:
    load_dotenv()  # 개발환경에서는 기본 위치의 .env 파일을 로드

class ExcelConverterPage(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 소스 엑셀 파일 선택을 위한 텍스트 필드
        self.input_file_field = StyledTextField("소스 엑셀 파일을 선택해주세요.")
        self.input_file_field.set_value(self.settings.get("excel_input_file", ""))
        self.input_file_button = create_styled_button("소스 엑셀 파일 선택", self.select_input_file)
        
        # 타겟 엑셀 파일 선택을 위한 텍스트 필드 및 버튼
        self.target_file_field = StyledTextField("타겟 엑셀 파일을 선택해주세요.")
        self.target_file_field.set_value(self.settings.get("excel_target_file", ""))
        self.target_file_button = create_styled_button("타겟 엑셀 파일 선택", self.select_target_file)
        
        self.convert_button = create_styled_button("변환 및 업데이트", self.convert_excel)

        # 상태 메시지 라벨
        self.status_label = QLabel("")

        # 레이아웃 구성
        layout.addWidget(self.input_file_field)
        layout.addWidget(self.input_file_button)
        layout.addWidget(self.target_file_field)
        layout.addWidget(self.target_file_button)
        layout.addWidget(self.convert_button)
        layout.addWidget(self.status_label)
        layout.addStretch()
        self.setLayout(layout)

    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "소스 엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.input_file_field.set_value(file_path)
            self.settings["excel_input_file"] = file_path
            save_settings(self.settings)

    def select_target_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "타겟 엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.target_file_field.set_value(file_path)
            self.settings["excel_target_file"] = file_path
            save_settings(self.settings)

    def convert_excel(self):
        input_file = self.input_file_field.get_value()
        target_file = self.target_file_field.get_value()

        if not input_file:
            QMessageBox.warning(self, "경고", "소스 엑셀 파일을 선택해주세요.")
            self.status_label.setText("소스 파일 경로가 필요합니다.")
            return

        if not target_file:
            QMessageBox.warning(self, "경고", "타겟 엑셀 파일을 선택해주세요.")
            self.status_label.setText("타겟 파일 경로가 필요합니다.")
            return

        # 소스 파일은 헤더가 포함된 엑셀로 읽음 (원본 데이터의 열 제목이 첫 행에 있다고 가정)
        try:
            source_df = pd.read_excel(input_file, header=0)
        except Exception as e:
            self.status_label.setText(f"소스 파일 읽기 오류: {str(e)}")
            QMessageBox.critical(self, "오류", f"소스 파일 읽기 중 오류 발생: {str(e)}")
            return

        # 타겟 파일은 이미 존재하는 경우, 첫 행이 열 제목임
        if os.path.exists(target_file):
            try:
                target_df = pd.read_excel(target_file, header=0)
            except Exception as e:
                self.status_label.setText(f"타겟 파일 읽기 오류: {e}")
                return
        else:
            QMessageBox.warning(self, "경고", "타겟 파일이 존재하지 않습니다. 먼저 타겟 파일을 준비해주세요.")
            self.status_label.setText("타겟 파일이 필요합니다.")
            return

        # 매핑 딕셔너리: 원본 열 이름 -> 타겟 열 이름
        mapping = {
            "접수번호": "접수번호",
            "기술분야": "기술분야",
            "상점명": "상점명",
            "진행상태": "진행상태",
            "계약예산금액": "계약예산금액",
            "국비금액": "국비금액",
            "본인부담금액": "본인부담금액",
            "사업자번호": "사업자번호",
            "대표자명": "대표자명",
            "담당자명": "담당자명",
            "담당자핸드폰": "담당자핸드폰",
            "담당자연락처": "담당자연락처",
            "담당자이메일": "담당자이메일",
            "우편번호": "우편번호",
            "기본주소": "기본주소",
            "상세주소": "상세 주소",
            "신청일자": "신청일자"
        }

        # 확인: 소스에 필요한 모든 열이 존재하는지 체크
        missing_cols = [col for col in mapping if col not in source_df.columns]
        if missing_cols:
            self.status_label.setText(f"소스 파일에 누락된 열: {missing_cols}")
            return

        # 대상 타겟 엑셀의 열 순서(형식)를 그대로 유지
        target_columns = target_df.columns.tolist()

        # '접수번호'를 기준으로 업데이트를 수행
        primary_key = "접수번호"

        # index 설정 (문자열 기준으로 업데이트를 쉽게 하기 위해)
        target_df[primary_key] = target_df[primary_key].astype(str)
        source_df[primary_key] = source_df[primary_key].astype(str)

        updates = []

        # for each row in source_df, update target_df if 접수번호 exists, else append as new row
        for _, src_row in source_df.iterrows():
            key = src_row[primary_key]
            # Check if key exists in target_df
            mask = target_df[primary_key] == key
            if mask.any():
                update_details = []
                # For each mapped column, compare the target value and source value
                for src_col, tgt_col in mapping.items():
                    if tgt_col in target_columns:
                        current_val = target_df.loc[mask, tgt_col].iloc[0]
                        new_val = src_row[src_col]
                        if str(current_val) != str(new_val):
                            update_details.append(f"{tgt_col}: {current_val} -> {new_val}")
                            target_df.loc[mask, tgt_col] = new_val
                if update_details:
                    updates.append(f"접수번호: {key}\n내역: {', '.join(update_details)}")
            else:
                # 신규 건: 새 행 생성 (타겟 파일의 전체 컬럼에 대해 기본값은 그대로 유지)
                new_row = {col: None for col in target_columns}
                for src_col, tgt_col in mapping.items():
                    if tgt_col in target_columns:
                        new_row[tgt_col] = src_row[src_col]
                new_row_df = pd.DataFrame([new_row])
                target_df = pd.concat([target_df, new_row_df], ignore_index=True)
                updates.append(f"접수번호: {key}\n내역: 신규 행 추가")

        if updates:
            message = "스마트상점 업데이트 발생\n" + "\n\n".join(updates)
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            if token and chat_id:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": chat_id, "text": message}
                try:
                    requests.post(url, data=payload)
                except Exception as e:
                    print(f"Telegram 알림 전송 오류: {e}")
            else:
                print("Telegram 설정이 env 파일에 없습니다.")

        try:
            target_df.to_excel(target_file, index=False)
            self.status_label.setText(f"성공적으로 업데이트됨: {target_file}")
            QMessageBox.information(self, "완료", "변환 및 업데이트가 완료되었습니다.")
        except Exception as e:
            self.status_label.setText(f"파일 저장 중 오류: {e}")
            QMessageBox.critical(self, "오류", f"파일 저장 중 오류 발생: {str(e)}")