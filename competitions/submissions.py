import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, SpaceHardware
from huggingface_hub.utils._errors import EntryNotFoundError

from competitions.enums import SubmissionStatus
from competitions.errors import AuthenticationError, PastDeadlineError, SubmissionError, SubmissionLimitError
from competitions.utils import token_information, team_file_api, user_token_api, server_manager


@dataclass
class Submissions:
    competition_id: str
    competition_type: str
    submission_limit: str
    hardware: str
    end_date: datetime
    token: str

    def _verify_submission(self, bytes_data):
        return True

    def _num_subs_today(self, todays_date, team_submission_info):
        todays_submissions = 0
        for sub in team_submission_info["submissions"]:
            submission_datetime = sub["datetime"]
            submission_date = submission_datetime.split(" ")[0]
            if submission_date == todays_date:
                todays_submissions += 1
        return todays_submissions

    def _is_submission_allowed(self, team_id):
        todays_date = datetime.now()
        if todays_date > self.end_date:
            raise PastDeadlineError("Competition has ended.")

        todays_date = todays_date.strftime("%Y-%m-%d")
        team_submission_info = self._download_team_submissions(team_id)

        if len(team_submission_info["submissions"]) == 0:
            team_submission_info["submissions"] = []

        todays_submissions = self._num_subs_today(todays_date, team_submission_info)
        if todays_submissions >= self.submission_limit:
            return False
        return True

    def _increment_submissions(
        self,
        team_id,
        user_id,
        submission_id,
        submission_comment,
        submission_repo=None,
        space_id=None,
    ):
        if submission_repo is None:
            submission_repo = ""
        if space_id is None:
            space_id = ""
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)
        datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # here goes all the default stuff for submission
        team_submission_info["submissions"].append(
            {
                "datetime": datetime_now,
                "submission_id": submission_id,
                "submission_comment": submission_comment,
                "submission_repo": submission_repo,
                "space_id": space_id,
                "submitted_by": user_id,
                "status": SubmissionStatus.PENDING.value,
                "selected": False,
                "public_score": {},
                "private_score": {},
                "server_url": server_manager.get_next_server(),
                "hardware": self.hardware,
            }
        )
        # count the number of times user has submitted today
        todays_date = datetime.now().strftime("%Y-%m-%d")
        todays_submissions = self._num_subs_today(todays_date, team_submission_info)
        self._upload_team_submissions(team_id, team_submission_info)
        return todays_submissions

    def _upload_team_submissions(self, team_id, team_submission_info):
        team_submission_info_json = json.dumps(team_submission_info, indent=4)
        team_submission_info_json_bytes = team_submission_info_json.encode("utf-8")
        team_submission_info_json_buffer = io.BytesIO(team_submission_info_json_bytes)
        api = HfApi(token=self.token)
        api.upload_file(
            path_or_fileobj=team_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def _download_team_submissions(self, team_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)
        return team_submission_info

    def update_selected_submissions(self, user_token, selected_submission_ids):
        current_datetime = datetime.now()
        if current_datetime > self.end_date:
            raise PastDeadlineError("Competition has ended.")

        team_id = team_file_api.get_team_info(user_token)["id"]
        team_submission_info = self._download_team_submissions(team_id)

        for sub in team_submission_info["submissions"]:
            if sub["submission_id"] in selected_submission_ids:
                sub["selected"] = True
            else:
                sub["selected"] = False

        self._upload_team_submissions(team_id, team_submission_info)

    def _get_team_subs(self, team_id, private=False):
        try:
            team_submissions_info = self._download_team_submissions(team_id)
        except EntryNotFoundError:
            return pd.DataFrame()
        submissions_df = pd.DataFrame(team_submissions_info["submissions"])

        if len(submissions_df) == 0:
            return pd.DataFrame()

        if not private:
            submissions_df = submissions_df.drop(columns=["private_score"])

        submissions_df = submissions_df.sort_values(by="datetime", ascending=False)
        submissions_df = submissions_df.reset_index(drop=True)

        # stringify public_score column
        submissions_df["public_score"] = submissions_df["public_score"].apply(json.dumps)

        if private:
            submissions_df["private_score"] = submissions_df["private_score"].apply(json.dumps)

        submissions_df["status"] = submissions_df["status"].apply(lambda x: SubmissionStatus(x).name)

        return submissions_df

    def _get_user_info(self, user_token):
        user_info = token_information(token=user_token)
        if "error" in user_info:
            raise AuthenticationError("Invalid token")

        # if user_info["emailVerified"] is False:
        #     raise AuthenticationError("Please verify your email on Hugging Face Hub")
        return user_info

    def my_submissions(self, user_token):
        current_date_time = datetime.now()
        private = False
        if current_date_time >= self.end_date:
            private = True
        team_id = team_file_api.get_team_info(user_token)["id"]
        if not team_id:
            return pd.DataFrame()
        return self._get_team_subs(team_id, private=private)

    def _create_submission(self, team_id: str):
        api = HfApi(token=self.token)

        if api.file_exists(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            repo_type="dataset",
        ):
            return

        team_submission_info = {}
        team_submission_info["id"] = team_id
        team_submission_info["submissions"] = []
        team_submission_info_json = json.dumps(team_submission_info, indent=4)
        team_submission_info_json_bytes = team_submission_info_json.encode("utf-8")
        team_submission_info_json_buffer = io.BytesIO(team_submission_info_json_bytes)

        api.upload_file(
            path_or_fileobj=team_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )
        return team_id

    def new_submission(self, user_token, uploaded_file, submission_comment):
        # verify token
        user_info = self._get_user_info(user_token)
        submission_id = str(uuid.uuid4())
        user_id = user_info["id"]
        team_id = team_file_api.get_team_info(user_token)["id"]
        self._create_submission(team_id)

        # check if team can submit to the competition
        if self._is_submission_allowed(team_id) is False:
            raise SubmissionLimitError("Submission limit reached")

        if self.competition_type == "generic":
            bytes_data = uploaded_file.file.read()
            # verify file is valid
            if not self._verify_submission(bytes_data):
                raise SubmissionError("Invalid submission file")

            file_extension = uploaded_file.filename.split(".")[-1]
            # upload file to hf hub
            api = HfApi(token=self.token)
            api.upload_file(
                path_or_fileobj=bytes_data,
                path_in_repo=f"submissions/{team_id}-{submission_id}.{file_extension}",
                repo_id=self.competition_id,
                repo_type="dataset",
            )
            submissions_made = self._increment_submissions(
                team_id=team_id,
                user_id=user_id,
                submission_id=submission_id,
                submission_comment=submission_comment,
            )
            return self.submission_limit - submissions_made

        user_api = HfApi(token=user_token)
        submission_id = user_api.model_info(repo_id=uploaded_file).sha + "__" + submission_id
        competition_organizer = self.competition_id.split("/")[0]
        space_id = f"{competition_organizer}/comp-{submission_id}"

        user_token_api.put(team_id, user_token)
        submissions_made = self._increment_submissions(
            team_id=team_id,
            user_id=user_id,
            submission_id=submission_id,
            submission_comment=submission_comment,
            submission_repo=uploaded_file,
            space_id=space_id,
        )
        remaining_submissions = self.submission_limit - submissions_made
        return remaining_submissions
