
using System.Text;

namespace ElincloudServer.Common
{
    public class TextCompareUtil
    {
        public string Compare(string textA, string textB, TextCompareMode mode = TextCompareMode.Result)
        {
            StringBuilder text = new StringBuilder();
            if (mode == TextCompareMode.Result)
            {
                text.Append(textB);
            }
            else
            {
                TextCompareItem[] diffs = TextComparer.Compare(textA, textB);
                int textLength = 0;
                int pos = 0;

                int progressCount = diffs.Length + 1;

                for (int n = 0; n < diffs.Length; n++)
                {
                    TextCompareItem it = diffs[n];

                    #region write unchanged chars

                    if ((pos < it.StartB) && (pos < textB.Length))
                    {
                        int min = Math.Min(it.StartB, textB.Length);
                        int len = min - pos;
                        string content = textB.Substring(pos, len);
                        if (content != string.Empty)
                        {
                            text.Append(content);
                            //SetTextBlockDefaultTextStyle(tb, content);
                        }
                        textLength += len;
                        pos += len;
                    }

                    #endregion

                    #region write deleted chars

                    if (mode == TextCompareMode.Detail)
                    {
                        if (it.deletedA > 0)
                        {

                            string deleteStr = textA.Substring(it.StartA, it.deletedA);
                            text.Append(string.Format("<span style=\"text-decoration-line: line-through; background-color:rgb(221,160,221)\">{0}</span>", deleteStr));
                            //SetTextBlockDeleteTextStyle(tb, deleteStr);
                            textLength += it.deletedA;

                        }
                    }

                    #endregion

                    #region write inserted chars

                    if (pos < it.StartB + it.insertedB)
                    {
                        string insertStr = textB.Substring(pos, it.insertedB);
                        text.Append(string.Format("<span style=\"background-color:rgb(144,238,144)\">{0}</span>", insertStr));
                        //SetTextBlockUpdateTextStyle(tb, textB.Substring(pos, it.insertedB));
                        textLength += it.insertedB;
                        pos += it.insertedB;
                    }

                    #endregion
                }

                #region write rest of unchanged chars

                if (pos < textB.Length)
                {
                    int len = textB.Length - pos;
                    string content = textB.Substring(pos, len);
                    if (content != string.Empty)
                    {
                        text.Append(content);
                        //SetTextBlockDefaultTextStyle(tb, content);
                        textLength += len;
                        pos += len;
                    }
                }

                #endregion

            }
            return text.ToString();
        }

        public double CalcMatchPercent(string textA, string textB)
        {
            List<MatchPoint> points = TextComparer.Match(textA, textB);
            double total = (double)points.Sum(p => p.Weight);
            double matched = (double)points.Sum(p => p.IsMatched ? p.Weight : 0);
            return Math.Round((matched / total) * 100, 4);
        }

        public double CalcSimilarityPercent(string textA, string textB)
        {
            int maxLenth = textA.Length > textB.Length ? textA.Length : textB.Length;
            int val = TextComparer.Distance(textA, textB);
            return Math.Round((double)val / maxLenth * 100, 4);
        }
    }
}
