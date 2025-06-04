import datetime
import os
import threading
import time
import io
from typing import Dict, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger
from pydantic import BaseModel
from requests.exceptions import RequestException
from filelock import FileLock, Timeout as FileLockTimeout

from competitions import __version__, utils
from competitions.errors import AuthenticationError, PastDeadlineError, SubmissionError, SubmissionLimitError
from competitions.info import CompetitionInfo
from competitions.leaderboard import Leaderboard
from competitions.oauth import attach_oauth
from competitions.runner import JobRunner
from competitions.submissions import Submissions
from competitions.text import SUBMISSION_SELECTION_TEXT, SUBMISSION_TEXT
from competitions.utils import team_file_api, submission_api, leaderboard_api
from competitions.enums import SubmissionStatus


HF_TOKEN = os.environ.get("HF_TOKEN", None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ID = os.environ.get("COMPETITION_ID")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/tmp/model")
START_DATE = os.environ.get("START_DATE", "2000-12-31")
DISABLE_PUBLIC_LB = int(os.environ.get("DISABLE_PUBLIC_LB", 0))

disable_progress_bars()

try:
    REQUIREMENTS_FNAME = hf_hub_download(
        repo_id=COMPETITION_ID,
        filename="requirements.txt",
        token=HF_TOKEN,
        repo_type="dataset",
    )
except EntryNotFoundError:
    REQUIREMENTS_FNAME = None

if REQUIREMENTS_FNAME:
    logger.info("Uninstalling and installing requirements")
    utils.uninstall_requirements(REQUIREMENTS_FNAME)
    utils.install_requirements(REQUIREMENTS_FNAME)


class LeaderboardRequest(BaseModel):
    lb: str


class UpdateSelectedSubmissionsRequest(BaseModel):
    submission_ids: str


class UpdateTeamNameRequest(BaseModel):
    new_team_name: str


def run_job_runner():
    job_runner = JobRunner(
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        output_path=OUTPUT_PATH,
    )
    job_runner.run()


def start_job_runner_thread():
    thread = threading.Thread(target=run_job_runner)
    # thread.daemon = True
    thread.start()
    return thread


def watchdog(job_runner_thread):
    while True:
        if not job_runner_thread.is_alive():
            logger.warning("Job runner thread stopped. Restarting...")
            job_runner_thread = start_job_runner_thread()
        time.sleep(10)


job_runner_thread = start_job_runner_thread()
watchdog_thread = threading.Thread(target=watchdog, args=(job_runner_thread,))
watchdog_thread.daemon = True
watchdog_thread.start()


app = FastAPI()
attach_oauth(app)

static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates_path = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)


@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    """
    This function is used to render the HTML file
    :param request:
    :return:
    """
    if HF_TOKEN is None:
        return HTTPException(status_code=500, detail="HF_TOKEN is not set.")
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    context = {
        "request": request,
        "logo": competition_info.logo_url,
        "competition_type": competition_info.competition_type,
        "version": __version__,
        "rules_available": competition_info.rules is not None,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/login_status", response_class=JSONResponse)
def use_oauth(request: Request, user_token: str = Depends(utils.user_authentication)):
    result = {
        "is_login": False,
        "is_admin": False,
        "is_registered": False,
    }

    comp_org = COMPETITION_ID.split("/")[0]
    if user_token:
        result["is_login"] = True
        result["is_admin"] = utils.is_user_admin(user_token, comp_org)
        result["is_registered"] = utils.team_file_api.get_team_info(user_token) is not None
    return {"response": result}


@app.get("/logout", response_class=HTMLResponse)
def user_logout(request: Request):
    """Endpoint that logs out the user (e.g. delete cookie session)."""

    if "oauth_info" in request.session:
        request.session.pop("oauth_info", None)

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    context = {
        "request": request,
        "logo": competition_info.logo_url,
        "competition_type": competition_info.competition_type,
        "__version__": __version__,
        "rules_available": competition_info.rules is not None,
    }

    return templates.TemplateResponse("index.html", context)


@app.get("/competition_info", response_class=JSONResponse)
def get_comp_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.competition_desc
    resp = {"response": info}
    return resp


@app.get("/dataset_info", response_class=JSONResponse)
def get_dataset_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.dataset_desc
    resp = {"response": info}
    return resp


@app.get("/rules", response_class=JSONResponse)
def get_rules(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    if competition_info.rules is not None:
        return {"response": competition_info.rules}
    return {"response": "No rules available."}


@app.get("/submission_info", response_class=JSONResponse)
def get_submission_info(request: Request):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    info = competition_info.submission_desc
    resp = {"response": info}
    return resp


@app.post("/leaderboard", response_class=JSONResponse)
def fetch_leaderboard(_: str = Depends(utils.user_authentication)):
    df = leaderboard_api.get_leaderboard()
    if len(df) == 0:
        return {"response": "No teams yet. Why not make a submission?"}
    resp = {"response": df.to_markdown(index=False)}
    return resp


@app.post("/my_submissions", response_class=JSONResponse)
def my_submissions(request: Request, user_token: str = Depends(utils.user_authentication)):
    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    if user_token is None:
        return {
            "response": {
                "submissions": "",
                "submission_text": SUBMISSION_TEXT.format(competition_info.submission_limit),
                "error": "**Invalid token. Please login.**",
                "team_name": "",
            }
        }
    sub = Submissions(
        end_date=competition_info.end_date,
        submission_limit=competition_info.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=competition_info.competition_type,
        hardware=competition_info.hardware,
    )
    try:
        subs = sub.my_submissions(user_token)
    except AuthenticationError:
        return {
            "response": {
                "submissions": "",
                "submission_text": SUBMISSION_TEXT.format(competition_info.submission_limit),
                "error": "**Invalid token. Please login.**",
                "team_name": "",
            }
        }
    subs = subs.to_dict(orient="records")
    error = ""
    if len(subs) == 0:
        error = "**You have not made any submissions yet.**"
        subs = ""
    submission_text = SUBMISSION_TEXT.format(competition_info.submission_limit)
    submission_selection_text = SUBMISSION_SELECTION_TEXT.format(competition_info.selection_limit)

    team_name = utils.team_file_api.get_team_info(user_token)["name"]

    resp = {
        "response": {
            "submissions": subs,
            "submission_text": submission_text + submission_selection_text,
            "error": error,
            "team_name": team_name,
        }
    }
    return resp


@app.post("/new_submission", response_class=JSONResponse)
def new_submission(
    hub_model: str = Form(...),
    submission_comment: str = Form(None),
    user_token: str = Depends(utils.user_authentication),
):
    if submission_comment is None:
        submission_comment = ""

    if user_token is None:
        return {"response": "Invalid token. Please login."}

    todays_date = datetime.datetime.now()
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    if todays_date < start_date:
        comp_org = COMPETITION_ID.split("/")[0]
        if not utils.is_user_admin(user_token, comp_org):
            return {"response": "Competition has not started yet!"}

    team_id = team_file_api.get_team_info(user_token)["id"]

    lock = FileLock(f"./submission_lock/{team_id}.lock", blocking=False)
    try:
        with lock:
            if submission_api.exists_submission_info(team_id) and submission_api.count_by_status(team_id, [SubmissionStatus.QUEUED, SubmissionStatus.PENDING, SubmissionStatus.PROCESSING]) > 0:
                return {"response": "Another submission is being processed. Please wait a moment."}
            competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
            sub = Submissions(
                end_date=competition_info.end_date,
                submission_limit=competition_info.submission_limit,
                competition_id=COMPETITION_ID,
                token=HF_TOKEN,
                competition_type=competition_info.competition_type,
                hardware=competition_info.hardware,
            )
            if competition_info.competition_type == "script":
                resp = sub.new_submission(user_token, hub_model, submission_comment)
                return {"response": f"Success! You have {resp} submissions remaining today."}
            return {"response": "Invalid competition type"}
    except RequestException:
        return {"response": "Hugging Face Hub is unreachable, please try again later"}
    except AuthenticationError:
        return {"response": "Invalid token"}
    except PastDeadlineError:
        return {"response": "Competition has ended"}
    except SubmissionError:
        return {"response": "Invalid submission file"}
    except SubmissionLimitError:
        return {"response": "Submission limit reached"}
    except FileLockTimeout:
        return {"response": "Another submission is being processed. Please wait a moment."}


@app.post("/update_selected_submissions", response_class=JSONResponse)
def update_selected_submissions(
    request: Request, body: UpdateSelectedSubmissionsRequest, user_token: str = Depends(utils.user_authentication)
):
    submission_ids = body.submission_ids

    if user_token is None:
        return {"success": False, "error": "Invalid token, please login."}

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)
    sub = Submissions(
        end_date=competition_info.end_date,
        submission_limit=competition_info.submission_limit,
        competition_id=COMPETITION_ID,
        token=HF_TOKEN,
        competition_type=competition_info.competition_type,
        hardware=competition_info.hardware,
    )
    submission_ids = submission_ids.split(",")
    submission_ids = [s.strip() for s in submission_ids]
    if len(submission_ids) > competition_info.selection_limit:
        return {
            "success": False,
            "error": f"Please select at most {competition_info.selection_limit} submissions.",
        }
    sub.update_selected_submissions(user_token=user_token, selected_submission_ids=submission_ids)
    return {"success": True, "error": ""}


@app.post("/update_team_name", response_class=JSONResponse)
def update_team_name(
    request: Request, body: UpdateTeamNameRequest, user_token: str = Depends(utils.user_authentication)
):
    new_team_name = body.new_team_name

    if user_token is None:
        return {"success": False, "error": "Invalid token. Please login."}

    if str(new_team_name).strip() == "":
        return {"success": False, "error": "Team name cannot be empty."}

    try:
        utils.team_file_api.update_and_create_team_name(user_token, new_team_name)
        return {"success": True, "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/admin/comp_info", response_class=JSONResponse)
def admin_comp_info(request: Request, user_token: str = Depends(utils.user_authentication)):
    comp_org = COMPETITION_ID.split("/")[0]
    user_is_admin = utils.is_user_admin(user_token, comp_org)
    if not user_is_admin:
        return {"response": "You are not an admin."}, 403

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)

    markdowns = {
        "competition_desc": competition_info.competition_desc,
        "rules": competition_info.rules,
        "submission_desc": competition_info.submission_desc,
        "dataset_desc": competition_info.dataset_desc,
    }
    if markdowns["rules"] is None:
        markdowns["rules"] = "No rules available."

    config = {
        "SUBMISSION_LIMIT": competition_info.submission_limit,
        "SELECTION_LIMIT": competition_info.selection_limit,
        "END_DATE": competition_info.end_date.strftime("%Y-%m-%d"),
        "EVAL_HIGHER_IS_BETTER": competition_info.eval_higher_is_better,
        "SUBMISSION_COLUMNS": competition_info.submission_columns_raw,
        "SUBMISSION_ID_COLUMN": competition_info.submission_id_col,
        "LOGO": competition_info.logo_url,
        "COMPETITION_TYPE": competition_info.competition_type,
        "EVAL_METRIC": competition_info.metric,
        "SUBMISSION_ROWS": competition_info.submission_rows,
        "TIME_LIMIT": competition_info.time_limit,
        "DATASET": competition_info.dataset,
        "SUBMISSION_FILENAMES": competition_info.submission_filenames,
        "SCORING_METRIC": competition_info.scoring_metric,
        "HARDWARE": competition_info.hardware,
    }

    return {"response": {"config": config, "markdowns": markdowns}}


@app.post("/admin/update_comp_info", response_class=JSONResponse)
def update_comp_info(data: Dict[str, Any] = Body(...), user_token: str = Depends(utils.user_authentication)):
    comp_org = COMPETITION_ID.split("/")[0]
    user_is_admin = utils.is_user_admin(user_token, comp_org)
    if not user_is_admin:
        return {"response": "You are not an admin."}, 403

    competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=HF_TOKEN)

    config = data["config"]
    markdowns = data["markdowns"]

    valid_keys = [
        "SUBMISSION_LIMIT",
        "SELECTION_LIMIT",
        "END_DATE",
        "EVAL_HIGHER_IS_BETTER",
        "SUBMISSION_COLUMNS",
        "SUBMISSION_ID_COLUMN",
        "LOGO",
        "COMPETITION_TYPE",
        "EVAL_METRIC",
        "SUBMISSION_ROWS",
        "TIME_LIMIT",
        "DATASET",
        "SUBMISSION_FILENAMES",
        "SCORING_METRIC",
        "HARDWARE",
    ]

    for key in config:
        if key not in valid_keys:
            return {"success": False, "error": f"Invalid key: {key}"}

    try:
        competition_info.update_competition_info(config, markdowns, HF_TOKEN)
    except Exception as e:
        logger.error(e)
        return {"success": False}, 500

    return {"success": True}


@app.get("/register_template_file")
def register_example_file():
    file_path = os.path.join(BASE_DIR, "static", "Competition Participants Information Collection Template.xlsx")
    return FileResponse(file_path, media_type="application/octet-stream", filename="Competition Participants Information Collection Template.xlsx")


@app.post("/register")
def register(
    teamname: str = Form(..., max_length=20),
    file: UploadFile = Form(...),
    email: str = Form(..., max_length=100),
    user_token: str = Depends(utils.user_authentication)
):
    if user_token is None:
        return {"success": False, "response": "Please login."}

    file_bytes = file.file.read()
    file_stream = io.BytesIO(file_bytes)
    file_stream.seek(0)

    utils.team_file_api.create_team(
        user_token,
        teamname,
        file_stream,
        email
    )

    return {"success": True, "response": "Team created successfully."}
