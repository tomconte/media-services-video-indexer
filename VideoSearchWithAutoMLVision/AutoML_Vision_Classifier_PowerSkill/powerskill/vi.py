import json
import logging
import requests

from azure.identity import DefaultAzureCredential


class VideoIndexerAPI():
    def __init__(self, avam_location, avam_subscription, avam_resource_group, avam_account_id, avam_account_name):
        self.avam_location = avam_location
        self.avam_subscription = avam_subscription
        self.avam_resource_group = avam_resource_group
        self.avam_account_id = avam_account_id
        self.avam_account_name = avam_account_name
        self.access_token = None

    def get_access_token(self):
        """
        Here we get an access token for the video indexer instance
        :return:
        """
        logging.info('Getting video indexer access token...')

        credential = DefaultAzureCredential()
        arm_token = credential.get_token('https://management.azure.com/.default')

        params = {
            'permissionType': 'Contributor',
            'scope': 'Account'
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + arm_token.token
        }

        access_token_req = requests.post(
            'https://management.azure.com/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.VideoIndexer/accounts/{acc_id}/generateAccessToken?api-version=2021-11-10-preview'.format(
                sub=self.avam_subscription,
                rg=self.avam_resource_group,
                acc_id=self.avam_account_name
            ),
            data=json.dumps(params),
            headers=headers
        )

        access_token = access_token_req.json()['accessToken']
        logging.info('Access Token successfully retrieved')
        self.access_token = access_token
        return access_token

    def get_thumbnail(self, video_id, thumbnail_id):
        """
        Get a thumbnail from the video
        :param video_id: Id of the video
        :param thumbnail_id: Id of the thumbnail
        :return: The image
        """
        logging.info('Getting video thumbnail..')

        headers = {
            'Authorization': 'Bearer ' + self.access_token
        }

        thumbnail_req = requests.get(
            'https://api.videoindexer.ai/{loc}/Accounts/{acc_id}/videos/{vid_id}/Thumbnails/{thumb_id}'.format(
                loc=self.avam_location,
                acc_id=self.avam_account_id,
                vid_id=video_id,
                thumb_id=thumbnail_id

            ),
            headers=headers
        )

        logging.info('Thumbnail: {}'.format(thumbnail_req))
        return thumbnail_req

    def get_video_artifacts(self, video_id):
        """
        Here we download all thumbnails for the video so that we can run
        inference on the keyframes
        :param video_id: Id of the video
        :return: A zip of downloaded artifacts
        """
        print('Getting video artifacts..')

        headers = {
            'Authorization': 'Bearer ' + self.access_token
        }

        artifacts_req = requests.get(
            'https://api.videoindexer.ai/{loc}/Accounts/{acc_id}/videos/{vid_id}/ArtifactUrl?type={artifact_type}'.format(
                loc=self.avam_location,
                acc_id=self.avam_account_id,
                vid_id=video_id,
                artifact_type='KeyframesThumbnails'

            ),
            headers=headers
        )

        logging.info('KeyFrame Thumbnail: {}'.format(artifacts_req))
        return artifacts_req

    def list_videos(self):
        print('Getting videos..')

        headers = {
            'Authorization': 'Bearer ' + self.access_token
        }

        list_videos_req = requests.get(
            'https://api.videoindexer.ai/{loc}/Accounts/{acc_id}/videos'.format(
                loc=self.avam_location,
                acc_id=self.avam_account_id
            ),
            headers=headers
        ).json()

        logging.info('Videos: {}'.format(list_videos_req))
        return list_videos_req
