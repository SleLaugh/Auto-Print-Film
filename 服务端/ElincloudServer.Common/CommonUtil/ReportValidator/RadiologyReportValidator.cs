using System.Text.RegularExpressions;
using ElincloudServer.Entity;

namespace ElincloudServer.Common
{
    public class RadiologyReportValidator
    {
        public List<ValidationResult> ValidateReport(PatientMetadata patient, string reportText, List<ValidationRuleInfo> rules)
        {
            var errors = new List<ValidationResult>();
            var normalizedText = PreprocessText(reportText);

            List<ValidationRuleInfo> subRulesErrorKeywords = rules.FindAll(p => p.ValidationType == ValidationTypes.错别字);

            foreach (var rule in subRulesErrorKeywords)
            {
                foreach (var keyword in rule.Keywords)
                {
                    if (normalizedText.Contains(keyword))
                    {
                        errors.Add(new ValidationResult()
                        {
                            ErrorType = rule.ValidationType.ToString(),
                            ErrorMsg = string.Format(rule.ErrorMessage, keyword)
                        });
                    }
                }
            }

            List<ValidationRuleInfo> subRulesOthers = rules.FindAll(p => p.ValidationType != ValidationTypes.错别字);

            foreach (var rule in subRulesOthers)
            {
                foreach (var keyword in rule.Keywords)
                {
                    if (ContainsKeywordWithContext(normalizedText, keyword, rule.Condition.GetNeedWarning(patient)))
                    {
                        errors.Add(new ValidationResult()
                        {
                            ErrorType = rule.ValidationType.ToString(),
                            ErrorMsg = string.Format(rule.ErrorMessage, keyword)
                        });
                    }
                }
            }

            return errors;
        }

        private string PreprocessText(string text)
        {
            //// 统一全角半角、去除标点、转换为小写
            //return Regex.Replace(text.Normalize(), @"[\p{P}\s]", " ")
            //           .ToLowerInvariant()
            //           .Replace("　", " ");
            return text.Normalize().ToLowerInvariant().Replace("　", " ");
        }

        private bool ContainsKeywordWithContext(string text, string keyword, bool needWarning)
        {
            MedicalTextMatcher matcher = new MedicalTextMatcher();
            //外面对text做了ToLowerInvariant处理，这里对关键字做同样的处理
            keyword = keyword.ToLowerInvariant();
            // 如果包含否定词则不计入错误
            MatchStatus status = matcher.AnalyzeReport(text, keyword);
            if (status == MatchStatus.KeywordNotFound)
            {
                //没有匹配到关键词，直接返回false
                return false;
            }
            else//匹配到关键词
            {
                // 当匹配到异常词组时（如女性患者出现男性特征）
                // 且不包含否定词时返回true
                if (needWarning && status == MatchStatus.FindingConfirmed) return true;

                // 当不需要反向检查时（如男性患者出现男性特征）
                // 且包含否定词时返回true
                if (!needWarning && status == MatchStatus.FindingNegated) return true;
            }

            return false;
        }
    }
}
