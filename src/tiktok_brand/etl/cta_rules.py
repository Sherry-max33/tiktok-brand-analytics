"""CTA detection via regex patterns (logic in code; YAML holds keywords only)."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern

PURCHASE_CTA_PATTERNS: List[str] = [
    r"\bshop\s+now\b",
    r"\bshop\s+(?:the\s+)?(?:look|collection|drop|edit|range|style|release)\b",
    r"\bshop\s+(?:online|today|here)\b",
    r"\bbuy\s+now\b",
    r"\bbuy\s+(?:it|them|yours|here|today)\b",
    r"\border\s+now\b",
    r"\border\s+(?:today|here|yours)\b",
    r"\bplace\s+(?:an?|your)\s+order\b",
    r"\bcheck\s*out\s+now\b",
    r"\bcheckout\s+now\b",
    r"\bget\s+(?:it|them|yours)\s+now\b",
    r"\bget\s+(?:it|them|yours)\b",
    r"\bget\s+your\s+(?:pair|copy|set)\b",
    r"\bgrab\s+(?:it|them|yours)\b",
    r"\bgrab\s+your\s+(?:pair|copy|set)\b",
    r"\bclaim\s+(?:it|yours|your\s+\w+)\b",
    r"\bsecure\s+(?:it|them|yours|your\s+(?:pair|spot))\b",
    r"\badd\s+(?:it|them|this)\s+to\s+(?:your\s+)?(?:cart|bag)\b",
    r"\badd\s+to\s+(?:cart|bag)\b",
    r"\bin\s+your\s+(?:cart|bag)\b",
    r"\bpre[\s-]?order\s+now\b",
    r"\bpre[\s-]?order\s+(?:here|today|yours)\b",
    r"\breserve\s+(?:now|yours|your\s+pair)\b",
    r"\bshop\s+(?:via|through|on)\s+tiktok\s+shop\b",
    r"\btap\s+(?:the\s+)?(?:product|shop|shopping)\s+link\b",
    r"\btap\s+(?:the\s+)?(?:yellow\s+)?(?:basket|cart|bag)\b",
    r"\bclick\s+(?:the\s+)?(?:product|shop|shopping)\s+link\b",
]

ENGAGEMENT_CTA_PATTERNS: List[str] = [
    r"\bcomment\s+(?:below|your|if|with)\b",
    r"\bleave\s+(?:a|your)\s+comment\b",
    r"\bdrop\s+(?:a|an|your)\s+(?:comment|answer|thoughts?|emoji)\b",
    r"\blet\s+us\s+know\b",
    r"\btell\s+us\b",
    r"\btell\s+me\b",
    r"\bwhat\s+do\s+you\s+think\b",
    r"\bwhat(?:'s|\s+is)\s+your\s+(?:favorite|favourite|pick|choice|opinion)\b",
    r"\bwhich\s+(?:one|pair|color|colour|style|look)\b",
    r"\bchoose\s+your\s+(?:favorite|favourite|pick)\b",
    r"\bvote\s+(?:below|now|for)\b",
    r"\banswer\s+(?:below|in\s+the\s+comments)\b",
    r"\blike\s+(?:this|the\s+video|if)\b",
    r"\btap\s+(?:the\s+)?like\s+button\b",
    r"\bdouble[\s-]?tap\b",
    r"\bgive\s+(?:this|it)\s+a\s+like\b",
    r"\bfollow\s+(?:us|me|for|to)\b",
    r"\bfollow\s+along\b",
    r"\bstay\s+tuned\b",
    r"\bstay\s+connected\b",
    r"\bdon(?:'|’)?t\s+miss\b",
    r"\bturn\s+on\s+(?:your\s+)?notifications\b",
    r"\bshare\s+(?:this|it|with)\b",
    r"\bsend\s+(?:this|it)\s+to\b",
    r"\bpass\s+(?:this|it)\s+on\b",
    r"\btag\s+(?:a|your|someone|the)\b",
    r"\bmention\s+(?:a|your|someone)\b",
    r"\btag\s+someone\s+who\b",
    r"\bsave\s+(?:this|it|for)\b",
    r"\bbookmark\s+(?:this|it)\b",
    r"\bkeep\s+this\s+for\s+later\b",
    r"\bjoin\s+(?:the|our)\s+(?:challenge|conversation|trend)\b",
    r"\btry\s+(?:this|it|the\s+challenge)\b",
    r"\bshow\s+us\b",
    r"\buse\s+(?:this\s+)?(?:sound|hashtag|filter|effect)\b",
    r"\bstitch\s+(?:this|with)\b",
    r"\bduet\s+(?:this|with)\b",
]

DISCOVERY_TRAFFIC_CTA_PATTERNS: List[str] = [
    r"\blink\s+in\s+(?:the|our|my)\s+bio\b",
    r"\blink\s+is\s+in\s+(?:the|our|my)\s+bio\b",
    r"\blinkinbio\b",
    r"\bbio\s+link\b",
    r"\bcheck\s+(?:the|our|my)\s+bio\b",
    r"\bvisit\s+(?:the|our|my)\s+bio\b",
    r"\bhead\s+to\s+(?:the|our|my)\s+bio\b",
    r"\bclick\s+(?:the|this|our)\s+link\b",
    r"\btap\s+(?:the|this|our)\s+link\b",
    r"\bfollow\s+(?:the|this)\s+link\b",
    r"\buse\s+(?:the|this)\s+link\b",
    r"\bvisit\s+(?:the|this)\s+link\b",
    r"\bvisit\s+(?:our|the)\s+(?:website|site|store|shop|page|profile)\b",
    r"\bhead\s+to\s+(?:our|the)\s+(?:website|site|store|shop|page|profile)\b",
    r"\bgo\s+to\s+(?:our|the)\s+(?:website|site|store|shop|page|profile)\b",
    r"\bfind\s+out\s+more\s+(?:online|on\s+our\s+website)\b",
    r"\bdownload\s+(?:the|our)\s+app\b",
    r"\bopen\s+(?:the|our)\s+app\b",
    r"\blearn\s+more\b",
    r"\bfind\s+out\s+more\b",
    r"\bdiscover\s+more\b",
    r"\bexplore\s+(?:more|the|our)\b",
    r"\bread\s+more\b",
    r"\bsee\s+more\b",
    r"\bwatch\s+(?:more|the\s+full)\b",
    r"\bcheck\s+(?:it|this|them)\s+out\b",
    r"\bcheck\s+out\s+(?:the|our|this|these)\b",
    r"\bsee\s+(?:the|our)\s+(?:full|latest|new)\b",
    r"\bsign\s+up\b",
    r"\bregister\s+(?:now|today|here)\b",
    r"\bjoin\s+(?:now|today|us|the\s+waitlist|our\s+community)\b",
    r"\bsubscribe\s+(?:now|today|here|to)\b",
    r"\benter\s+(?:now|today|here|the\s+giveaway)\b",
]

PROMO_PATTERNS: List[str] = [
    r"\bon\s+sale\b",
    r"\b(?:summer|winter|holiday|flash|exclusive)\s+sale\b",
    r"\bsale\s+(?:now|starts|ends|alert)\b",
    r"\bdiscount(?:ed)?\b",
    r"\b\d{1,3}%\s+off\b",
    r"\bsave\s+\$?\d+\b",
    r"\bsave\s+\d{1,3}%\b",
    r"\bpromo\s+code\b",
    r"\bdiscount\s+code\b",
    r"\buse\s+code\b",
    r"\bcode\s+[a-z0-9]+\b",
    r"\blimited[\s-]?time\s+offer\b",
    r"\bspecial\s+offer\b",
    r"\bexclusive\s+offer\b",
    r"\bfree\s+shipping\b",
    r"\bfree\s+delivery\b",
    r"\bbuy\s+one\s+get\s+one\b",
    r"\bbogo\b",
    r"\bwhile\s+supplies\s+last\b",
    r"\bwhile\s+stocks?\s+last\b",
    r"\blimited\s+quantit(?:y|ies)\b",
    r"\bends?\s+(?:today|tonight|soon)\b",
    r"\blast\s+chance\b",
    r"\bavailable\s+now\b",
    r"\bout\s+now\b",
    r"\bdrops?\s+(?:today|tomorrow|soon|now)\b",
    r"\bcoming\s+soon\b",
    r"\bnew\s+(?:drop|release|collection|launch)\b",
]


def _compile(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_COMPILED = {
    "purchase": _compile(PURCHASE_CTA_PATTERNS),
    "engagement": _compile(ENGAGEMENT_CTA_PATTERNS),
    "discovery_traffic": _compile(DISCOVERY_TRAFFIC_CTA_PATTERNS),
    "promo": _compile(PROMO_PATTERNS),
}


def matches_any(text: str, patterns: List[Pattern[str]]) -> bool:
    if not text or not patterns:
        return False
    return any(p.search(text) for p in patterns)


def detect_cta_flags(caption: str) -> Dict[str, bool]:
    text = (caption or "").lower()
    purchase = matches_any(text, _COMPILED["purchase"])
    engagement = matches_any(text, _COMPILED["engagement"])
    discovery = matches_any(text, _COMPILED["discovery_traffic"])
    promo = matches_any(text, _COMPILED["promo"])
    return {
        "has_purchase_cta": purchase,
        "has_engagement_cta": engagement,
        "has_discovery_traffic_cta": discovery,
        "has_promo_language": promo,
        "has_cta": purchase or engagement or discovery,
    }
