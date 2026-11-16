using System.Text;
using System.Text.RegularExpressions;
using ElincloudServer.Entity;

namespace ElincloudServer.Common
{
    public class MedicalTextMatcher
    {

        // 综合匹配引擎
        public MatchStatus AnalyzeReport(string text, string keyword)
        {
            // 预处理：统一字符格式
            text = PreprocessText(text);

            // 核心匹配流程
            bool hasKeyword = Regex.IsMatch(text, keyword);
            bool isNegated = CheckMedicalNegation(text, keyword);

            return EvaluateStatus(hasKeyword, isNegated);
        }

        // 医学否定判断逻辑（综合两种算法优点）
        private bool CheckMedicalNegation(string text, string keyword)
        {
            // 双重验证机制
            return Regex.IsMatch(text, BuildStrictNegationPattern(keyword))
                || Regex.IsMatch(text, BuildContextNegationPattern(keyword));
        }

        // 模式1：严格否定规则（精确匹配）
        private string BuildStrictNegationPattern(string keyword)
        {
            // 特征：否定词紧邻关键词（0-2个中文字符间隔）
            return string.Format(@"(?<negation>未[见示]?|无|不)[\u4e00-\u9fa5]{{0,2}}?{0}(?![\u4e00-\u9fa5]*[但然而])", keyword);
        }

        // 模式2：上下文否定规则（带距离限制）
        private string BuildContextNegationPattern(string keyword)
        {
            // 特征：否定词与关键词间隔≤8字符，且中间无转折词
            return string.Format(@"(?<negation>未|无)[\u4e00-\u9fa5]{{1,8}}?{0}(?<!未增强|未分化)", keyword);
        }

        // 状态决策树
        private MatchStatus EvaluateStatus(bool hasKeyword, bool isNegated)
        {
            //return hasKeyword switch {
            //    true when isNegated => MatchStatus.FindingNegated,
            //    true => MatchStatus.FindingConfirmed,
            //    _ => MatchStatus.KeywordNotFound
            //};
            if (hasKeyword)
            {
                if (isNegated)
                {
                    return MatchStatus.FindingNegated;
                }
                else
                {
                    return MatchStatus.FindingConfirmed;
                }
            }
            else
            {
                return MatchStatus.KeywordNotFound;
            }

        }

        // 文本标准化处理
        private string PreprocessText(string text)
        {
            return Regex.Replace(text.Normalize(NormalizationForm.FormKC), @"[\s\u3000]+", "");
        }
    }

}
