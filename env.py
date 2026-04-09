import math
import numpy as np
import pandas as pd
from faker import Faker
from typing import Tuple, Dict, Any
from models import Observation

fake = Faker()

class DataCleaningEnv:
    def __init__(self):
        self.task_id = None
        self.gold_df = None
        self.current_df = None
        self.previous_accuracy = 0.0
        self.step_count = 0
        self.max_steps = 15
        self.done = False
        self.validation_errors = []

    def reset(self, task_id: str) -> Dict[str, Any]:
        self.task_id = task_id
        self.step_count = 0
        self.previous_accuracy = 0.0
        self.done = False
        self.validation_errors = []
        if task_id == 'fix_types':
            self.gold_df, self.current_df = self._make_fix_types(120, seed=42)
        elif task_id == 'normalize_dedupe':
            self.gold_df, self.current_df = self._make_normalize_dedupe(100, seed=43)
        elif task_id == 'full_pipeline':
            self.gold_df, self.current_df = self._make_full_pipeline(200, seed=44)
        else:
            raise ValueError('Unknown task_id')
        return self._get_obs()

    def state(self) -> Observation:
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.current_df is None or self.gold_df is None:
            return self._get_obs(), -0.05, False, {'reason': 'environment not reset', 'invalid': True}

        if self.done:
            return self._get_obs(), 0.0, True, {'reason': 'episode already done'}

        name = action.get('name')
        params = action.get('params', {})
        invalid = False
        reward = 0.0
        self.step_count += 1

        try:
            if name in ('cast', 'cast_type'):
                col = params['column']
                dtype = params.get('dtype', params.get('target_type'))
                self._action_cast(col, dtype)
            elif name == 'normalize_dates':
                col = params['column']
                self._action_normalize_dates(col)
            elif name in ('dedupe', 'drop_duplicates'):
                subset = params.get('subset', None)
                self._action_dedupe(subset)
            elif name == 'fill_missing':
                col = params['column']
                strategy = params.get('strategy', 'mean')
                self._action_fill_missing(col, strategy)
            elif name == 'replace':
                self._action_replace(
                    column=params.get('column'),
                    old_value=params.get('old_value'),
                    new_value=params.get('new_value'),
                )
            elif name == 'clamp_outliers':
                col = params['column']
                low = params.get('low', None)
                high = params.get('high', None)
                self._action_clamp(col, low, high)
            elif name == 'submit':
                # submission: check accuracy and end
                pass
            else:
                invalid = True
        except Exception as e:
            invalid = True
            self.validation_errors.append(f'action_error: {str(e)}')

        current_accuracy = self._compute_accuracy()
        delta = current_accuracy - self.previous_accuracy
        # Allow negative reward for regressive actions
        reward = max(-1.0, min(1.0, delta))
        if invalid:
            reward = reward - 0.05
        if self.step_count >= self.max_steps and not self.done:
            self.validation_errors.append('max_steps_reached')

        info = {
            'accuracy': float(current_accuracy),
            'grader_score': float(current_accuracy),
            'previous_accuracy': float(self.previous_accuracy),
            'invalid': bool(invalid),
            'validation_errors': self._to_python_json_types(self.validation_errors[-10:]),
        }
        self.previous_accuracy = current_accuracy

        if name == 'submit':
            if math.isclose(current_accuracy, 1.0):
                reward = min(1.0, reward + 1.0)
            self.done = True

        if math.isclose(current_accuracy, 1.0) or self.step_count >= self.max_steps:
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, info

    def _get_obs(self) -> Observation:
        if self.current_df is None:
            preview = []
            accuracy = 0.0
            markdown_preview = ''
            null_counts = {}
            validation_errors = self._to_python_json_types(self.validation_errors[-10:])
        else:
            preview_raw = self.current_df.head(5).to_dict(orient='records')
            preview = self._to_python_json_types(preview_raw)
            accuracy = float(self._compute_accuracy())
            markdown_preview = self._build_markdown_preview(self.current_df.head(5))
            null_counts = self._to_python_json_types(self.current_df.isna().sum().to_dict())
            validation_errors = self._collect_validation_errors()
        return Observation(
            dataframe_preview=preview,
            markdown_preview=str(markdown_preview),
            null_counts={str(k): int(v) for k, v in null_counts.items()},
            validation_errors=[str(v) for v in validation_errors],
            accuracy=float(accuracy),
            step_count=int(self.step_count),
            steps_remaining=max(0, int(self.max_steps - self.step_count)),
        )

    def _to_python_json_types(self, obj):
        if isinstance(obj, dict):
            return {self._to_python_json_types(k): self._to_python_json_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_python_json_types(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_python_json_types(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return obj

    def _build_markdown_preview(self, df: pd.DataFrame) -> str:
        if df.empty:
            return '| empty |\n|---|\n| no rows |'
        columns = [str(c) for c in df.columns]
        header = '| ' + ' | '.join(columns) + ' |'
        separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
        body = []
        for _, row in df.iterrows():
            values = []
            for c in df.columns:
                value = self._to_python_json_types(row[c])
                values.append('' if value is None else str(value))
            body.append('| ' + ' | '.join(values) + ' |')
        return '\n'.join([header, separator] + body)

    def _collect_validation_errors(self):
        errors = list(self.validation_errors[-10:])
        if self.current_df is None:
            return errors

        if 'age' in self.current_df.columns:
            age_series = pd.to_numeric(self.current_df['age'], errors='coerce')
            invalid_age = int(((age_series < 0) | (age_series > 120)).fillna(False).sum())
            if invalid_age > 0:
                errors.append(f'age_out_of_range={invalid_age}')

        if 'income' in self.current_df.columns:
            income_series = pd.to_numeric(
                self.current_df['income'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce'
            )
            negative_income = int((income_series < 0).fillna(False).sum())
            if negative_income > 0:
                errors.append(f'negative_income={negative_income}')

        return self._to_python_json_types(errors[-10:])

    # --- Data generators ---
    def _make_fix_types(self, n, seed=0):
        np.random.seed(seed)
        rows = []
        gold_rows = []
        for i in range(n):
            val = round(np.random.uniform(0, 5000), 2)
            name = fake.name()
            is_active_val = bool(np.random.randint(0, 2))
            dt = fake.date_between(start_date='-5y', end_date='today')
            
            # store messy variants
            if np.random.rand() < 0.5:
                messy_amount = f"{val:,.2f}"
            else:
                messy_amount = f"${val:,.2f}"
                
            if np.random.rand() < 0.5:
                messy_active = str(is_active_val).lower()
            else:
                messy_active = "1" if is_active_val else "0"
                
            messy_date = dt.strftime('%m/%d/%Y')  # string format
                
            rows.append({
                'id': i, 
                'name': name, 
                'is_active': messy_active, 
                'joined_date': messy_date, 
                'amount': messy_amount
            })
            gold_rows.append({
                'id': i, 
                'name': name, 
                'is_active': is_active_val, 
                'joined_date': dt.isoformat(), 
                'amount': val
            })
        return pd.DataFrame(gold_rows), pd.DataFrame(rows)

    def _make_normalize_dedupe(self, n, seed=0):
        np.random.seed(seed)
        gold_rows = []
        rows = []
        for i in range(n):
            canonical_name = fake.name()
            dt = fake.date_between(start_date='-2y', end_date='today')
            # varied formats
            if np.random.rand() < 0.5:
                sdt = dt.strftime('%m/%d/%y')
            else:
                sdt = dt.isoformat()
            gold_rows.append({'id': i, 'name': canonical_name, 'date': dt.isoformat()})
            rows.append({'id': i, 'name': canonical_name, 'date': sdt})
        # add duplicates
        dup_count = max(1, int(n * 0.1))
        for j in range(dup_count):
            rows.append(rows[j].copy())
        return pd.DataFrame(gold_rows).reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)

    def _make_full_pipeline(self, n, seed=0):
        np.random.seed(seed)
        rows = []
        gold_rows = []
        for i in range(n):
            name = fake.name()
            age = int(np.random.normal(40, 12))
            income = round(max(0, np.random.normal(60000, 20000)), 2)
            if np.random.rand() < 0.05:
                # extreme outlier
                income = income * 100
            if np.random.rand() < 0.1:
                age = None
            # logic violation
            if np.random.rand() < 0.02:
                age = 250
            clean_age = 40 if age is None else max(0, min(120, int(age)))
            gold_rows.append({'id': i, 'name': name, 'age': clean_age, 'income': income})
            # create messy variants in fields that can be fixed with supported actions
            income_str = f"${income:,.2f}"
            rows.append({'id': i, 'name': name, 'age': age, 'income': income_str})
        df_gold = pd.DataFrame(gold_rows)
        df_cur = pd.DataFrame(rows)
        return df_gold.reset_index(drop=True), df_cur.reset_index(drop=True)

    # --- Actions implementations ---
    def _action_cast(self, column, dtype):
        if column not in self.current_df.columns:
            raise ValueError('column missing')
        if dtype in ('float', 'float64'):
            self.current_df[column] = self.current_df[column].astype(str).str.replace(r'[\$,]', '', regex=True)
            self.current_df[column] = pd.to_numeric(self.current_df[column], errors='coerce').astype(float)
        elif dtype in ('int', 'int64'):
            self.current_df[column] = self.current_df[column].astype(str).str.replace(r'[\$,]', '', regex=True)
            self.current_df[column] = pd.to_numeric(self.current_df[column], errors='coerce').fillna(0).astype(int)
        elif dtype in ('bool', 'boolean'):
            mapping = {'true': True, 'false': False, '1': True, '0': False, 't': True, 'f': False}
            self.current_df[column] = self.current_df[column].astype(str).str.lower().map(mapping).fillna(False)
        elif dtype in ('datetime', 'date'):
            self.current_df[column] = pd.to_datetime(self.current_df[column], errors='coerce')
        elif dtype in ('str', 'string'):
            self.current_df[column] = self.current_df[column].astype(str)
        else:
            raise ValueError('unsupported dtype')

    def _action_normalize_dates(self, column):
        if column not in self.current_df.columns:
            raise ValueError('column missing')
        self.current_df[column] = pd.to_datetime(self.current_df[column], errors='coerce').dt.strftime('%Y-%m-%d')

    def _action_dedupe(self, subset=None):
        if subset is None:
            self.current_df = self.current_df.drop_duplicates()
        else:
            self.current_df = self.current_df.drop_duplicates(subset=subset)
        self.current_df = self.current_df.reset_index(drop=True)

    def _action_fill_missing(self, column, strategy='mean'):
        if column not in self.current_df.columns:
            raise ValueError('column missing')
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(self.current_df[column]):
                val = pd.to_numeric(self.current_df[column], errors='coerce').mean()
                self.current_df[column] = pd.to_numeric(self.current_df[column], errors='coerce').fillna(val)
            else:
                self.current_df[column] = self.current_df[column].fillna('')
        elif strategy == 'ffill':
            self.current_df[column] = self.current_df[column].fillna(method='ffill')
        else:
            self.current_df[column] = self.current_df[column].fillna(strategy)

    def _action_clamp(self, column, low=None, high=None):
        if column not in self.current_df.columns:
            raise ValueError('column missing')
        self.current_df[column] = pd.to_numeric(self.current_df[column], errors='coerce')
        if low is not None:
            self.current_df.loc[self.current_df[column] < low, column] = low
        if high is not None:
            self.current_df.loc[self.current_df[column] > high, column] = high

    def _action_replace(self, column=None, old_value=None, new_value=None):
        if old_value is None:
            raise ValueError('old_value required for replace')
        if column is not None:
            if column not in self.current_df.columns:
                raise ValueError('column missing')
            self.current_df[column] = self.current_df[column].replace(old_value, new_value)
            return
        self.current_df = self.current_df.replace(old_value, new_value)

    # --- Metrics ---
    def _compute_accuracy(self) -> float:
        if self.current_df is None or self.gold_df is None:
            return 0.0
        # align shapes by reindexing current to gold by id if possible
        try:
            if 'id' in self.gold_df.columns and 'id' in self.current_df.columns:
                cur = self.current_df.set_index('id').reindex(self.gold_df['id']).reset_index()
            else:
                # fallback: align by position
                cur = self.current_df.copy()
                cur = cur.reindex(range(len(self.gold_df))).reset_index(drop=True)
        except Exception:
            cur = self.current_df.copy()
        # compare cellwise after stringifying normalized numbers/dates
        gold = self.gold_df.copy()
        def normalize_cell(x):
            if pd.isna(x):
                return '<<NA>>'
            if isinstance(x, (int, np.integer, float, np.floating)):
                return f"{x:.6f}"
            return str(x).strip()
        matches = 0
        total = 0
        for c in gold.columns:
            gcol = gold[c].astype(object).apply(normalize_cell).reset_index(drop=True)
            if c in cur.columns:
                ccol = cur[c].astype(object).apply(normalize_cell).reset_index(drop=True)
            else:
                ccol = pd.Series(['<<MISSING>>'] * len(gcol))
            if len(ccol) != len(gcol):
                ccol = ccol.reindex(range(len(gcol))).fillna('<<MISSING>>').reset_index(drop=True)
            comp = (gcol == ccol)
            matches += int(comp.sum())
            total += len(comp)
        if total == 0:
            return 0.01
        raw = matches / total
        return max(0.01, min(0.99, raw))


# If run as module for quick manual tests
if __name__ == '__main__':
    env = DataCleaningEnv()
    obs = env.reset('fix_types')
    print('Initial acc:', obs.accuracy)
    obs, r, done, info = env.step({'name': 'cast', 'params': {'column': 'amount', 'dtype': 'float'}})
    print('After cast acc:', obs.accuracy, 'reward', r, 'done', done)
