"""Regression checks for the two observed SSH monitoring failure modes."""
import subprocess,unittest
from unittest.mock import patch
from scripts import finish_auxiliary_evaluation,finish_coverage
WORKFLOWS=(finish_auxiliary_evaluation,finish_coverage)

class MonitorTests(unittest.TestCase):
    def test_command_timeout_is_retried(self):
        for workflow in WORKFLOWS:
            result=subprocess.CompletedProcess([],0,stdout='ready',stderr='')
            with patch.object(workflow.subprocess,'run',side_effect=[subprocess.TimeoutExpired('ssh',70),result]) as call,patch.object(workflow.time,'sleep'):
                self.assertEqual(workflow.run('get job example'),'ready')
                self.assertEqual(call.call_count,2)
    def test_ssh_failure_is_retried(self):
        for workflow in WORKFLOWS:
            results=[subprocess.CompletedProcess([],255,stdout='',stderr='timeout'),subprocess.CompletedProcess([],0,stdout='ready',stderr='')]
            with patch.object(workflow.subprocess,'run',side_effect=results),patch.object(workflow.time,'sleep'):
                self.assertEqual(workflow.run('get job example'),'ready')
    def test_exhausted_timeouts_are_connectivity_errors(self):
        for workflow in WORKFLOWS:
            with patch.object(workflow.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',70)),patch.object(workflow.time,'sleep'):
                with self.assertRaises(ConnectionError):workflow.run('get job example')
    def test_cluster_error_is_not_retried_as_connectivity(self):
        for workflow in WORKFLOWS:
            result=subprocess.CompletedProcess([],1,stdout='',stderr='NotFound')
            with patch.object(workflow.subprocess,'run',return_value=result) as call,patch.object(workflow.time,'sleep'):
                with self.assertRaises(RuntimeError):workflow.run('get job example')
                self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
