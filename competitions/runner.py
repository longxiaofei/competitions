import glob
import io
import json
import os
import time
import uuid
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from loguru import logger

from competitions.enums import SubmissionStatus
from competitions.info import CompetitionInfo
from competitions.utils import user_token_api, space_cleaner


@dataclass
class JobRunner:
    competition_id: str
    token: str
    output_path: str

    def __post_init__(self):
        self.competition_info = CompetitionInfo(competition_id=self.competition_id, autotrain_token=self.token)
        self.competition_id = self.competition_info.competition_id
        self.competition_type = self.competition_info.competition_type
        self.metric = self.competition_info.metric
        self.submission_id_col = self.competition_info.submission_id_col
        self.submission_cols = self.competition_info.submission_cols
        self.submission_rows = self.competition_info.submission_rows
        self.time_limit = self.competition_info.time_limit
        self.dataset = self.competition_info.dataset
        self.submission_filenames = self.competition_info.submission_filenames

    def _get_all_submissions(self) -> List[Dict[str, Any]]:
        submission_jsons = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            token=self.token,
            repo_type="dataset",
        )
        submission_jsons = glob.glob(os.path.join(submission_jsons, "submission_info/*.json"))
        all_submissions = []
        for _json_path in submission_jsons:
            with open(_json_path, "r", encoding="utf-8") as f:
                _json = json.load(f)
            team_id = _json["id"]
            for sub in _json["submissions"]:
                all_submissions.append(
                    {
                        "team_id": team_id,
                        "submission_id": sub["submission_id"],
                        "datetime": sub["datetime"],
                        "status": sub["status"],
                        "submission_repo": sub["submission_repo"],
                        "space_id": sub["space_id"],
                        "server_url": sub["server_url"],
                        "hardware": sub["hardware"],
                    }
                )
        return all_submissions


    def _get_pending_subs(self, submissions: List[Dict[str, Any]]) -> pd.DataFrame:
        pending_submissions = []
        for sub in submissions:
            if sub["status"] == SubmissionStatus.PENDING.value:
                pending_submissions.append(sub)
        if len(pending_submissions) == 0:
            return None
        logger.info(f"Found {len(pending_submissions)} pending submissions.")
        pending_submissions = pd.DataFrame(pending_submissions)
        pending_submissions["datetime"] = pd.to_datetime(pending_submissions["datetime"])
        pending_submissions = pending_submissions.sort_values("datetime")
        pending_submissions = pending_submissions.reset_index(drop=True)
        return pending_submissions

    def _get_server_active_count(self, submissions: List[Dict[str, Any]]) -> Dict[str, int]:
        server_active_count = defaultdict(int)
        for sub in submissions:
            if sub["status"] in {SubmissionStatus.PROCESSING.value, SubmissionStatus.QUEUED.value}:
                server_active_count[sub["server_url"]] += 1
        return server_active_count

    def _queue_submission(self, team_id, submission_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        for submission in team_submission_info["submissions"]:
            if submission["submission_id"] == submission_id:
                submission["status"] = SubmissionStatus.QUEUED.value
                break

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

    def mark_submission_failed(self, team_id, submission_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        for submission in team_submission_info["submissions"]:
            if submission["submission_id"] == submission_id:
                submission["status"] = SubmissionStatus.FAILED.value

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

    def _create_readme(self, project_name: str) -> str:
        _readme = "---\n"
        _readme += f"title: {project_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "---\n"
        return _readme

    def create_space(self, team_id, submission_id, submission_repo, space_id, server_url, hardware):
        user_token = user_token_api.get(team_id)

        api = HfApi(token=self.token)
        params = {
            "space_id": space_id,
            "client_space_id": space_id,
            "competition_id": self.competition_id,
            "team_id": team_id,
            "submission_id": submission_id,
            "output_path": self.output_path,
            "submission_repo": submission_repo,
            "time_limit": self.time_limit,
            "dataset": self.dataset,
            "submission_filenames": self.submission_filenames,
        }
        token_info_json = json.dumps(params, indent=4)
        token_info_json_bytes = token_info_json.encode("utf-8")
        token_info_json_buffer = io.BytesIO(token_info_json_bytes)

        api = HfApi(token=self.token)
        client_token = uuid.uuid4().hex + uuid.uuid4().hex

        api.upload_file(
            path_or_fileobj=token_info_json_buffer,
            path_in_repo=f"token_data_info/{client_token}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=hardware,
            private=True,
        )

        api.add_space_secret(repo_id=space_id, key="HUGSIM_API_TOKEN", value=client_token)
        api.add_space_secret(repo_id=space_id, key="HUGSIM_SERVER_HOST", value=server_url)
        client_code_local_dir = f"/tmp/data/client_repo/{space_id}"
        client_commits = api.list_repo_commits(submission_repo, repo_type="model")
        api.snapshot_download(
            repo_id=submission_repo,
            repo_type="model",
            revision=client_commits[0].commit_id,
            token=user_token,
            local_dir=client_code_local_dir,
            allow_patterns=["*"],
        )
        with open(f"{client_code_local_dir}/README.md", "w", encoding="utf-8") as f:
            f.write(self._create_readme(space_id))
        try:
            api.upload_folder(
                repo_id=space_id,
                repo_type="space",
                folder_path=client_code_local_dir,
            )
        finally:
            shutil.rmtree(client_code_local_dir, ignore_errors=True)
        self._queue_submission(team_id, submission_id)

    def run(self):
        cur = 0
        while True:
            time.sleep(5)
            if cur == 10000:
                cur = 0
            cur += 1
            all_submissions = self._get_all_submissions()

            # Clean up spaces every 100 iterations
            if cur % 100 == 0:
                for space in all_submissions:
                    if space["status"] == SubmissionStatus.QUEUED.value:
                        space_cleaner.clean_space(
                            space["space_id"],
                            space["team_id"],
                            space["submission_id"],
                        )

            pending_submissions = self._get_pending_subs(all_submissions)
            if pending_submissions is None:
                continue
            first_pending_sub = pending_submissions.iloc[0]
            server_active_count = self._get_server_active_count(all_submissions)

            if server_active_count[first_pending_sub["server_url"]] >= 1:
                continue
            try:
                self.create_space(first_pending_sub["team_id"], first_pending_sub["submission_id"], first_pending_sub["submission_repo"], first_pending_sub["space_id"], first_pending_sub["server_url"], first_pending_sub["hardware"])
            except Exception as e:
                logger.error(
                    f"Failed to create space for {first_pending_sub['submission_id']}: {e}"
                )
                # mark submission as failed
                self.mark_submission_failed(first_pending_sub['team_id'], first_pending_sub['submission_id'])
                try:
                    space_cleaner.delete_space(first_pending_sub["space_id"])
                except Exception as e:
                    logger.error(f"Failed to delete space {first_pending_sub['space_id']}: {e}")
                logger.error(f"Marked submission {first_pending_sub['submission_id']} as failed.")
                continue
