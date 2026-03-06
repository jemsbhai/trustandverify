"""trustandverify — Jac Interface Layer.

Thin walker that delegates the full verification pipeline to the Python
trustandverify library via jac_interop. All scoring, search, LLM calls,
and export happen in Python — the Jac layer provides the graph structure
and walker interface.

Usage:
    jac run trustgraph.jac "Is remote work more productive?"
    jac run trustgraph.jac "Is coffee healthy?" --claims 5 --format markdown
"""

import from trustandverify.jac_interop {
    jac_verify, jac_export
}
import json;
import sys;
import os;

# ──────────────────────────────────────────────
# Node Types
# ──────────────────────────────────────────────

node Query {
    has text: str;
    has created_at: str = "";
}

node VerifiedClaim {
    has text: str;
    has verdict: str = "no_evidence";
    has confidence: float = 0.0;
    has belief: float = 0.0;
    has disbelief: float = 0.0;
    has uncertainty: float = 1.0;
    has assessment: str = "";
}

node ReportNode {
    has report_id: str = "";
    has query_text: str = "";
    has summary: str = "";
    has data: dict = {};
}

# ──────────────────────────────────────────────
# Edge Types
# ──────────────────────────────────────────────

edge HasClaim {}
edge HasReport {}

# ──────────────────────────────────────────────
# Walker — delegates to trustandverify pipeline
# ──────────────────────────────────────────────

walker TrustVerify {
    has query_text: str = "";
    has num_claims: int = 0;
    has search_backend: str = "tavily";
    has llm_backend: str = "gemini";
    has storage_backend: str = "memory";
    has db_path: str = "trustandverify.db";
    has export_format: str = "jsonld";
    has enable_byzantine: bool = False;
    has verbose: bool = False;
    has result: dict = {};

    can run with entry {
        print(f"\n{'='*60}");
        print(f"  trustandverify — Jac Interface");
        print(f"  {self.search_backend} search · {self.llm_backend} LLM");
        print(f"{'='*60}");
        print(f"\n  Query: {self.query_text}\n");

        # Delegate the full pipeline to Python trustandverify
        self.result = jac_verify(
            self.query_text,
            search_backend=self.search_backend,
            llm_backend=self.llm_backend,
            num_claims=self.num_claims,
            enable_byzantine=self.enable_byzantine,
            enable_cache=True,
            storage_backend=self.storage_backend,
            db_path=self.db_path,
            verbose=self.verbose
        );

        # Build the graph from the result
        q = Query(text=self.query_text, created_at=self.result.get("created_at", ""));
        here ++> q;

        for cd in self.result.get("claims", []) {
            op = cd.get("opinion", {});
            vc = VerifiedClaim(
                text=cd["text"],
                verdict=cd.get("verdict", "no_evidence"),
                confidence=op.get("projected_probability", 0.0) if op else 0.0,
                belief=op.get("belief", 0.0) if op else 0.0,
                disbelief=op.get("disbelief", 0.0) if op else 0.0,
                uncertainty=op.get("uncertainty", 1.0) if op else 1.0,
                assessment=cd.get("assessment", "")
            );
            q +>: HasClaim :+> vc;
        }

        rn = ReportNode(
            report_id=self.result.get("id", ""),
            query_text=self.query_text,
            summary=self.result.get("summary", ""),
            data=self.result
        );
        q +>: HasReport :+> rn;

        # Print results
        print(f"\n{'='*60}");
        print(f"  VERIFICATION REPORT");
        print(f"{'='*60}\n");
        for i, cd in enumerate(self.result.get("claims", [])) {
            op = cd.get("opinion", {});
            p = op.get("projected_probability", 0.0) if op else 0.0;
            v = cd.get("verdict", "no_evidence");
            print(f"  {i+1}. [{v.upper()} P={p:.3f}] {cd['text']}");
            if cd.get("assessment") {
                print(f"     {cd['assessment'][:120]}");
            }
            print("");
        }
        conflicts = self.result.get("conflicts", []);
        if len(conflicts) > 0 {
            print("  CONFLICTS:");
            for c in conflicts {
                print(f"    - {c['claim_text']} (degree={c['conflict_degree']:.3f})");
            }
            print("");
        }
        print(f"  SUMMARY: {self.result.get('summary', '')}\n");

        # Export
        if self.export_format != "none" {
            ext_map = {"jsonld": ".jsonld", "markdown": ".md", "md": ".md", "html": ".html"};
            ext = ext_map.get(self.export_format, ".jsonld");
            output_path = f"output{ext}";
            jac_export(self.result, format=self.export_format, output_path=output_path);
            print(f"  Report saved to {output_path}");
        }

        print(f"{'='*60}\n");
    }
}

# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

with entry {
    query = "Is remote work more productive than office work?";
    num_claims = 0;
    search_backend = "tavily";
    llm_backend = "gemini";
    storage_backend = "memory";
    db_path = "trustandverify.db";
    export_format = "jsonld";
    enable_byzantine = False;
    verbose = False;

    # CLI args
    args = sys.argv[1:];
    flag_map = {
        "--claims": "num_claims",
        "--search": "search_backend",
        "--llm": "llm_backend",
        "--storage": "storage_backend",
        "--db": "db_path",
        "--format": "export_format",
    };
    i = 0;
    remaining_args = [];
    while i < len(args) {
        if args[i] == "--byzantine" {
            enable_byzantine = True;
            i += 1;
        } elif args[i] == "--verbose" or args[i] == "-v" {
            verbose = True;
            i += 1;
        } elif args[i] in flag_map {
            if i + 1 < len(args) {
                flag_name = flag_map[args[i]];
                val = args[i + 1];
                if flag_name == "num_claims" {
                    num_claims = int(val);
                } elif flag_name == "search_backend" {
                    search_backend = val;
                } elif flag_name == "llm_backend" {
                    llm_backend = val;
                } elif flag_name == "storage_backend" {
                    storage_backend = val;
                } elif flag_name == "db_path" {
                    db_path = val;
                } elif flag_name == "export_format" {
                    export_format = val;
                }
                i += 2;
            } else {
                i += 1;
            }
        } else {
            remaining_args.append(args[i]);
            i += 1;
        }
    }
    if len(remaining_args) > 0 {
        query = " ".join(remaining_args);
    }

    agent = TrustVerify(
        query_text=query,
        num_claims=num_claims,
        search_backend=search_backend,
        llm_backend=llm_backend,
        storage_backend=storage_backend,
        db_path=db_path,
        export_format=export_format,
        enable_byzantine=enable_byzantine,
        verbose=verbose
    );
    root spawn agent;
}
