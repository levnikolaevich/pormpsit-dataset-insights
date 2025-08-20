import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from "recharts";
import styles from "@/styles/DomainLabels.module.css";
import { DataFormatter, numberFormatter } from "@/lib/helpers";

function DomainLabels({ labels }) {
  const data = Object.entries(labels)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className={styles.domainLabels}>
      <h2>Domain labels</h2>
      <div className={styles.graphCont}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={DataFormatter} />
            <YAxis type="category" dataKey="name" width={150} />
            <Tooltip formatter={(value) => numberFormatter(value)} />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default DomainLabels;
