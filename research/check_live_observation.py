"""Explicit opt-in diagnosis of the app's real sensors. Never saves raw images."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',action='store_true')
    p.add_argument('--camera',action='store_true')
    p.add_argument('--goals',nargs='*',default=[])
    p.add_argument('--output',type=Path,default=Path('runs/live-observation-check/result.json'))
    a=p.parse_args()
    if not a.screen and not a.camera:p.error('Explicit --screen and/or --camera consent flags required')
    from local_agent.app import configure_dpi
    configure_dpi()
    from local_agent.live_observation import capture_foreground,capture_webcam,default_specialists,analyze_screen
    report={'at':time.time(),'raw_images_saved':False};image=None
    if a.screen:
        try:image=capture_foreground()
        except Exception as exc:report['screen_error']=str(exc)
    if a.camera:
        try:
            result=capture_webcam();result['size']=result['image'].size;result.pop('image').close()
            report['camera']=result
            print('Camera frame received; face_count='+str(result['face_count']),flush=True)
        except Exception as exc:
            report['camera_error']=str(exc);print('Camera error: '+str(exc),flush=True)
    if image:
        try:
            registry=default_specialists()
            report['screen']=analyze_screen(image,a.goals,[],registry)
            print('Screen read locally; characters='+str(len(report['screen']['visible_text']))+
                  '; roles='+','.join(x['role'] for x in report['screen']['specialists']),flush=True)
        except Exception as exc:report['screen_error']=str(exc);print('Screen error: '+str(exc),flush=True)
        finally:image.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('Local diagnostic saved: '+str(a.output),flush=True)
    if any(key.endswith('_error') for key in report):raise SystemExit(1)


if __name__=='__main__':main()
