import io
import json
import os
import shlex
import subprocess
import traceback
import threading
import uuid
import base64
import glob
from typing import Optional, Dict, Any, List
from collections import defaultdict

import requests
import pandas as pd
from fastapi import Request
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from loguru import logger
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from huggingface_hub import SpaceStage
from cachetools import cached, TTLCache

from competitions.enums import SubmissionStatus
from competitions.params import EvalParams

from . import HF_URL


USER_TOKEN = os.environ.get("USER_TOKEN")


def token_information(token):
    if token.startswith("hf_oauth"):
        _api_url = HF_URL + "/oauth/userinfo"
    else:
        _api_url = HF_URL + "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            _api_url,
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")

    if response.status_code != 200:
        logger.error(f"Failed to request whoami-v2 - {response.status_code}")
        raise Exception("Invalid token.")

    resp = response.json()
    user_info = {}

    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:
        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


def user_authentication(request: Request):
    auth_header = request.headers.get("Authorization")
    bearer_token = None

    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.split(" ")[1]

    if bearer_token:
        try:
            _ = token_information(token=bearer_token)
            return bearer_token
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None

    if USER_TOKEN is not None:
        try:
            _ = token_information(token=USER_TOKEN)
            return USER_TOKEN
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None

    if "oauth_info" in request.session:
        try:
            _ = token_information(token=request.session["oauth_info"]["access_token"])
            return request.session["oauth_info"]["access_token"]
        except Exception as e:
            request.session.pop("oauth_info", None)
            logger.error(f"Failed to verify token: {e}")
            return None

    return None


def user_authentication_dep(token, return_raw=False):
    if token.startswith("hf_oauth"):
        _api_url = HF_URL + "/oauth/userinfo"
    else:
        _api_url = HF_URL + "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            _api_url,
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")

    resp = response.json()
    if return_raw:
        return resp

    user_info = {}
    if "error" in resp:
        return resp
    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:

        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


def make_clickable_user(user_id):
    link = "https://huggingface.co/" + user_id
    return f'<a  target="_blank" href="{link}">{user_id}</a>'


def run_evaluation(params, local=False, wait=False):
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    params = EvalParams(**params)
    if not local:
        params.output_path = "/tmp/model"
    params.save(output_dir=params.output_path)
    cmd = [
        "python",
        "-m",
        "competitions.evaluate",
        "--config",
        os.path.join(params.output_path, "params.json"),
    ]

    cmd = [str(c) for c in cmd]
    logger.info(cmd)
    env = os.environ.copy()
    cmd = shlex.split(" ".join(cmd))
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid


def pause_space(params):
    if "SPACE_ID" in os.environ:
        if os.environ["SPACE_ID"].split("/")[-1].startswith("comp-"):
            logger.info("Pausing space...")
            api = HfApi(token=params.token)
            api.pause_space(repo_id=os.environ["SPACE_ID"])


def delete_space(params):
    if "SPACE_ID" in os.environ:
        if os.environ["SPACE_ID"].split("/")[-1].startswith("comp-"):
            logger.info("Deleting space...")
            api = HfApi(token=params.token)
            api.delete_repo(repo_id=os.environ["SPACE_ID"], repo_type="space")


def uninstall_requirements(requirements_fname):
    if os.path.exists(requirements_fname):
        # read the requirements.txt
        uninstall_list = []
        with open(requirements_fname, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("-"):
                    uninstall_list.append(line[1:])

        # create an uninstall.txt
        with open("uninstall.txt", "w", encoding="utf-8") as f:
            for line in uninstall_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "uninstall",
                "-r",
                "uninstall.txt",
                "-y",
            ],
        )
        pipe.wait()
        logger.info("Requirements uninstalled.")
        return


def install_requirements(requirements_fname):
    # check if params.project_name has a requirements.txt
    if os.path.exists(requirements_fname):
        # install the requirements using subprocess, wait for it to finish
        install_list = []

        with open(requirements_fname, "r", encoding="utf-8") as f:
            for line in f:
                # if line startswith - then skip but dont skip if line startswith --
                if line.startswith("-"):
                    if not line.startswith("--"):
                        continue
                install_list.append(line)

        with open("install.txt", "w", encoding="utf-8") as f:
            for line in install_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "install",
                "-r",
                "install.txt",
            ],
        )
        pipe.wait()
        logger.info("Requirements installed.")
        return
    logger.info("No requirements.txt found. Skipping requirements installation.")
    return


def is_user_admin(user_token, competition_organization):
    user_info = token_information(token=user_token)
    user_orgs = user_info.get("orgs", [])
    for org in user_orgs:
        if org == competition_organization:
            return True
    return False


class TeamAlreadyExistsError(Exception):
    """Custom exception for when a team already exists."""
    pass

class TeamFileApi:
    def __init__(self, hf_token: str, competition_id: str):
        self.hf_token = hf_token
        self.competition_id = competition_id
        self._lock = threading.Lock()

    def _get_team_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        user_team = hf_hub_download(
            repo_id=self.competition_id,
            filename="user_team.json",
            token=self.hf_token,
            repo_type="dataset",
        )

        with open(user_team, "r", encoding="utf-8") as f:
            user_team = json.load(f)

        if user_id not in user_team:
            return None

        team_id = user_team[user_id]

        team_metadata = hf_hub_download(
            repo_id=self.competition_id,
            filename="teams.json",
            token=self.hf_token,
            repo_type="dataset",
        )

        with open(team_metadata, "r", encoding="utf-8") as f:
            team_metadata = json.load(f)

        return team_metadata[team_id]

    def _create_team(self, user_id: str, team_name: str, team_file: Optional[io.BytesIO], email: str) -> str:
        with self._lock:
            user_team = hf_hub_download(
                repo_id=self.competition_id,
                filename="user_team.json",
                token=self.hf_token,
                repo_type="dataset",
            )
            with open(user_team, "r", encoding="utf-8") as f:
                user_team = json.load(f)

            team_metadata = hf_hub_download(
                repo_id=self.competition_id,
                filename="teams.json",
                token=self.hf_token,
                repo_type="dataset",
            )

            with open(team_metadata, "r", encoding="utf-8") as f:
                team_metadata = json.load(f)

            # create a new team, if user is not in any team
            team_id = str(uuid.uuid4())
            user_team[user_id] = team_id

            team_metadata[team_id] = {
                "id": team_id,
                "name": team_name,
                "members": [user_id],
                "leader": user_id,
                "email": email,
            }

            user_team_json = json.dumps(user_team, indent=4)
            user_team_json_bytes = user_team_json.encode("utf-8")
            user_team_json_buffer = io.BytesIO(user_team_json_bytes)

            team_metadata_json = json.dumps(team_metadata, indent=4)
            team_metadata_json_bytes = team_metadata_json.encode("utf-8")
            team_metadata_json_buffer = io.BytesIO(team_metadata_json_bytes)

            api = HfApi(token=self.hf_token)
            api.upload_file(
                path_or_fileobj=user_team_json_buffer,
                path_in_repo="user_team.json",
                repo_id=self.competition_id,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=team_metadata_json_buffer,
                path_in_repo="teams.json",
                repo_id=self.competition_id,
                repo_type="dataset",
            )

            if team_file is not None:
                resp = api.upload_file(
                    path_or_fileobj=team_file,
                    path_in_repo=f"team_datas/{team_id}.xlsx",
                    repo_id=self.competition_id,
                    repo_type="dataset",
                )
        return team_id
    
    def create_team(self, user_token: str, team_name: str, team_file: io.BytesIO, email: str) -> str:
        user_info = token_information(token=user_token)
        return self._create_team(user_info["id"], team_name, team_file, email)

    def get_team_info(self, user_token: str) -> Optional[Dict[str, Any]]:
        user_info = token_information(token=user_token)
        return self._get_team_info(user_info["id"])

    def update_team_name(self, user_token, new_team_name):
        user_info = token_information(token=user_token)
        user_id = user_info["id"]
        team_info = self._get_team_info(user_id)

        with self._lock:
            team_metadata = hf_hub_download(
                repo_id=self.competition_id,
                filename="teams.json",
                token=self.hf_token,
                repo_type="dataset",
            )
            with open(team_metadata, "r", encoding="utf-8") as f:
                team_metadata = json.load(f)

            team_metadata[team_info["id"]]["name"] = new_team_name
            team_metadata_json = json.dumps(team_metadata, indent=4)
            team_metadata_json_bytes = team_metadata_json.encode("utf-8")
            team_metadata_json_buffer = io.BytesIO(team_metadata_json_bytes)
            api = HfApi(token=self.hf_token)
            api.upload_file(
                path_or_fileobj=team_metadata_json_buffer,
                path_in_repo="teams.json",
                repo_id=self.competition_id,
                repo_type="dataset",
            )
        return new_team_name


team_file_api = TeamFileApi(
    os.environ.get("HF_TOKEN", None),
    os.environ.get("COMPETITION_ID"),
)


class UserTokenApi:
    def __init__(self, hf_token: str, key_base64: str, competition_id: str):
        self.hf_token = hf_token
        self.key = base64.b64decode(key_base64)
        self.competition_id = competition_id
    
    def _encrypt(self, text: str) -> str:
        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)
        encrypted_data = aesgcm.encrypt(nonce, text.encode(), None)
        return base64.b64encode(nonce + encrypted_data).decode()

    def _decrypt(self, encrypted_text: str) -> str:
        aesgcm = AESGCM(self.key)
        data = base64.b64decode(encrypted_text)
        nonce = data[:12]
        ciphertext = data[12:]
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()
    
    def put(self, team_id: str, user_token: str):
        encrypted_token = self._encrypt(user_token)
        api = HfApi(token=self.hf_token)
        api.upload_file(
            path_or_fileobj=io.BytesIO(encrypted_token.encode()),
            path_in_repo=f"team_user_tokens/{team_id}",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def get(self, team_id: str) -> Optional[str]:
        try:
            user_token = hf_hub_download(
                repo_id=self.competition_id,
                filename=f"team_user_tokens/{team_id}",
                token=self.hf_token,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error(f"Failed to download user token - {e}")
            return None

        with open(user_token, "r", encoding="utf-8") as f:
            encrypted_token = f.read()

        return self._decrypt(encrypted_token)


user_token_api = UserTokenApi(
    os.environ.get("HF_TOKEN", None),
    os.environ.get("USER_TOKEN_KEY_BASE64"),
    os.environ.get("COMPETITION_ID")
)


class ServerManager:
    def __init__(self, server_url_list: List[str]):
        self.server_url_list = server_url_list
        self._cur_index = 0
        self._lock = threading.Lock()
    
    def get_next_server(self) -> str:
        with self._lock:
            server_url = self.server_url_list[self._cur_index]
            self._cur_index = (self._cur_index + 1) % len(self.server_url_list)
        return server_url
    

server_manager = ServerManager(["https://xdimlab-hugsim-web-server-0.hf.space"])


class SubmissionApi:
    def __init__(self, hf_token: str, competition_id: str):
        self.hf_token = hf_token
        self.competition_id = competition_id
        self.api = HfApi(token=hf_token)
    
    def exists_submission_info(self, team_id: str) -> bool:
        """
        Check if submission info exists for a given team ID.
        Args:
            team_id (str): The team ID.
        Returns:
            bool: True if submission info exists, False otherwise.
        """
        return self.api.file_exists(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            repo_type="dataset",
        )

    def download_submission_info(self, team_id: str) -> Dict[str, Any]:
        """
        Download the submission info from Hugging Face Hub.
        Args:
            team_id (str): The team ID.
        Returns:
            Dict[str, Any]: The submission info.
        """
        submission_info_path = self.api.hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            repo_type="dataset",
        )
        with open(submission_info_path, 'r') as f:
            submission_info = json.load(f)
        
        return submission_info

    def upload_submission_info(self, team_id: str, user_submission_info: Dict[str, Any]):
        user_submission_info_json = json.dumps(user_submission_info, indent=4)
        user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
        user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)
        self.api.upload_file(
            path_or_fileobj=user_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def update_submission_data(self, team_id: str, submission_id: str, data: Dict[str, Any]):
        user_submission_info = self.download_submission_info(team_id)
        for submission in user_submission_info["submissions"]:
            if submission["submission_id"] == submission_id:
                submission.update(data)
                break
        self.upload_submission_info(team_id, user_submission_info)

    def update_submission_status(self, team_id: str, submission_id: str, status: int):
        self.update_submission_data(team_id, submission_id, {"status": status})
    
    def count_by_status(self, team_id: str, status_list: List[SubmissionStatus]) -> int:
        user_submission_info = self.download_submission_info(team_id)
        count = sum(1 for submission in user_submission_info["submissions"] if SubmissionStatus(submission["status"]) in status_list)
        return count


submission_api = SubmissionApi(
    hf_token=os.environ.get("HF_TOKEN", None),
    competition_id=os.environ.get("COMPETITION_ID")
)


class SpaceCleaner:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
        self.space_build_error_count = defaultdict(int)
    
    def delete_space(self, space_id: str):
        """Delete a space by its ID."""
        self.api.delete_repo(repo_id=space_id, repo_type="space")

    def clean_space(self, space_id: str, team_id: str, submission_id: str):
        try:
            space_info = self.api.space_info(repo_id=space_id)
        except RepositoryNotFoundError:
            submission_api.update_submission_status(
                team_id=team_id,
                submission_id=submission_id,
                data={"status": SubmissionStatus.FAILED.value, "error_message": "start space failed."},

            )
            return
        if space_info.runtime.stage == SpaceStage.BUILD_ERROR:
            self.space_build_error_count[space_id] += 1
            if self.space_build_error_count[space_id] >= 3:
                self.delete_space(space_id)
                submission_api.update_submission_status(
                    team_id=team_id,
                    submission_id=submission_id,
                    status=SubmissionStatus.FAILED.value
                )
            else:
                self.api.restart_space(repo_id=space_id)
            return
        
        if space_info.runtime.stage == SpaceStage.RUNTIME_ERROR:
            self.delete_space(space_id)
            submission_api.update_submission_status(
                team_id=team_id,
                submission_id=submission_id,
                status=SubmissionStatus.FAILED.value
            )
            return


space_cleaner = SpaceCleaner(
    os.environ.get("HF_TOKEN", None),
)


class LeaderboardApi:
    def __init__(self, hf_token: str, competition_id: str):
        self.hf_token = hf_token
        self.competition_id = competition_id
        self.api = HfApi(token=hf_token)

    @cached(cache=TTLCache(maxsize=1, ttl=300))
    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get the leaderboard for the competition.
        Returns:
            pd.DataFrame: The leaderboard as a DataFrame.
        """
        all_scores = self._get_all_scores()
        if not all_scores:
            return pd.DataFrame(columns=["team_id", "team_name", "rc", "hdscore"])

        df = pd.DataFrame(all_scores)
        df = df.sort_values(by=["hdscore", "rc"], ascending=[False, False])
        df = df.groupby("team_id").first()
        df = df.sort_values(by=["hdscore", "rc"], ascending=[False, False])
        df['rank'] = range(1, len(df) + 1)
        df.insert(0, 'rank', df.pop('rank'))
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_all_scores(self) -> List[Dict[str, Any]]:
        team_metadata = self.api.hf_hub_download(
            repo_id=self.competition_id,
            filename="teams.json",
            repo_type="dataset",
        )
        with open(team_metadata, "r", encoding="utf-8") as f:
            team_metadata = json.load(f)

        submission_jsons = self.api.snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            repo_type="dataset",
        )
        submission_jsons = glob.glob(os.path.join(submission_jsons, "submission_info/*.json"))
        all_scores = []
        for _json_path in submission_jsons:
            with open(_json_path, "r", encoding="utf-8") as f:
                _json = json.load(f)
            team_id = _json["id"]
            for sub in _json["submissions"]:
                if sub["status"] != SubmissionStatus.SUCCESS.value:
                    continue
                all_scores.append({
                    "team_id": team_id,
                    "team_name": team_metadata[team_id]["name"],
                    "rc": sub["score"]["rc"],
                    "hdscore": sub["score"]["hdscore"],
                })
        return all_scores


leaderboard_api = LeaderboardApi(
    hf_token=os.environ.get("HF_TOKEN", None),
    competition_id=os.environ.get("COMPETITION_ID")
)
