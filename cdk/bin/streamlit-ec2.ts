#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { StreamlitEc2Stack } from '../lib/streamlit-ec2-stack';

const app = new cdk.App();
const repoName = (app.node.tryGetContext('repoName') as string) ?? 'streamlit-viz';
const imageTag = (app.node.tryGetContext('imageTag') as string) ?? 'latest';
const containerPort = Number(app.node.tryGetContext('containerPort') ?? 8000);
const instanceType = (app.node.tryGetContext('instanceType') as string) ?? 't3.small';
const certificateArn = app.node.tryGetContext('certificateArn') as string | undefined;
const hostedZoneDomain = app.node.tryGetContext('hostedZoneDomain') as string | undefined;
const subdomain = app.node.tryGetContext('subdomain') as string | undefined;

new StreamlitEc2Stack(app, 'StreamlitEc2Stack', {
  repoName,
  imageTag,
  containerPort,
  instanceType,
  certificateArn,
  hostedZoneDomain,
  subdomain,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
