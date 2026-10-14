"""Shared, dependency-free semantic boundary between training and the app."""
LABELS = {
    'camera': ['screen_facing','away','unobservable'],
    'screen': ['code','document','terminal','video','other'],
    'context': ['direct','support','waiting','interrupted','unknown']}
APP_CATEGORIES = ['editor','browser','terminal','documents','meeting','other']
TASK_KINDS = ['general','coding','writing','research','admin']
FEATURES = (['camera_'+x for x in LABELS['camera']] + ['screen_'+x for x in LABELS['screen']]
            + ['camera_available','screen_available','camera_age','screen_age','idle']
            + ['app_'+x for x in APP_CATEGORIES] + ['task_'+x for x in TASK_KINDS])
VERSION = 'return-model-v1'
