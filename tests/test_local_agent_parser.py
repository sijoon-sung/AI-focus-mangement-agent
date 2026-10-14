import unittest
from local_agent.note_parser import duration, grounded_outline, source_lines
from local_agent.inbox import parse_notes


class ParserTests(unittest.TestCase):
    def test_natural_duration(self):
        for text,minutes in [('자소서 쓰기 1시간',60),('글쓰기 1시간 30분',90),('논문 [40분]',40)]:
            self.assertEqual(parse_notes(text)[0]['explicit_minutes'],minutes)
        for text in ['회의 30분 후','독서 1~2시간','회의 1시간 30분 후']:
            self.assertIsNone(duration(text)[1])

    def test_shared_subjects_source_and_time(self):
        sources=source_lines('보고서 Atlas\n자소서 쓰기 1시간\n시험 공부 A과목, 신호 및 시스템, SW')
        tasks=[dict(source_line=0,title='Atlas 보고서'),dict(source_line=1,title='자소서 쓰기')]
        tasks += [dict(source_line=2,title=t+' 시험 공부') for t in ['A과목','신호 및 시스템','SW']]
        result=grounded_outline(sources,tasks)
        self.assertEqual(len(parse_notes(result['outline'])),5)
        self.assertIn('자소서 쓰기 [60분]',result['outline'])
        self.assertEqual(result['notices'],[])

    def test_missing_name_or_line_uses_original(self):
        result=grounded_outline(['보고서 Atlas','메일 답장'],[dict(source_line=0,title='보고서 작성')])
        self.assertIn('Atlas',result['outline'])
        self.assertIn('메일 답장',result['outline'])
        self.assertEqual(len(result['notices']),2)

    def test_no_multiplying_group_time_or_inventing_minutes(self):
        result=grounded_outline(['공부 A, B 1시간'],[dict(source_line=0,title='A 공부'),dict(source_line=0,title='B 공부')])
        self.assertEqual(result['outline'].count('[60분]'),1)
        result=grounded_outline(['보고서'],[dict(source_line=0,title='보고서 90분')])
        self.assertEqual(result['outline'],'보고서')

    def test_bad_source_rejected(self):
        with self.assertRaises(ValueError):
            grounded_outline(['원본'],[dict(source_line=8,title='없는 작업')])
