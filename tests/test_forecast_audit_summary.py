import unittest
from scripts.summarize_forecast_audit import summarize

class ForecastAuditSummaryTests(unittest.TestCase):
    def records(self):
        return [dict(id=str(i),budget=b,action=str(a),predicted_entropy=float(a),posterior_entropy=float(a),probability_brier=0.,shared_nll=float(a),shared_correct=a==b,values_forecast_mse=0.,values_mean_mse=1.,masks_forecast_mse=0.,masks_mean_mse=0.) for i in range(60) for b in range(5) for a in range(b,5)]
    def test_perfect_ranking_and_forecast(self):
        out=summarize(self.records())
        self.assertEqual(out['candidate_records'],900)
        self.assertEqual(out['entropy_mae'],0)
        self.assertEqual(out['entropy_pair_concordance'],1)
        self.assertEqual(out['shared_score_pair_concordance'],1)
        self.assertEqual(out['shared_nll_regret'],0)
        self.assertEqual(out['selected_check_accuracy'],1)
    def test_reversed_forecasts_have_regret(self):
        rows=self.records()
        for r in rows:r['predicted_entropy']=-r['predicted_entropy']
        out=summarize(rows)
        self.assertEqual(out['entropy_pair_concordance'],0)
        self.assertEqual(out['shared_score_pair_concordance'],0)
        self.assertEqual(out['shared_nll_regret'],2)
    def test_missing_candidate_rejected(self):
        with self.assertRaises(AssertionError):summarize(self.records()[:-1])
