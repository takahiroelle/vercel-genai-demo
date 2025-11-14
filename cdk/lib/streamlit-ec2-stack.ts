import { RemovalPolicy, Stack, StackProps, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as elbv2Targets from 'aws-cdk-lib/aws-elasticloadbalancingv2-targets';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as route53Targets from 'aws-cdk-lib/aws-route53-targets';
import * as logs from 'aws-cdk-lib/aws-logs';

export interface StreamlitEc2StackProps extends StackProps {
  repoName: string;
  imageTag?: string;
  containerPort?: number;
  instanceType?: string;
  certificateArn?: string;
  hostedZoneDomain?: string;
  subdomain?: string;
}

export class StreamlitEc2Stack extends Stack {
  constructor(scope: Construct, id: string, props: StreamlitEc2StackProps) {
    super(scope, id, props);

    const port = props.containerPort ?? 8000;
    const imageTag = props.imageTag ?? 'latest';
    const instanceType = props.instanceType ?? 't3.small';

    const vpc = new ec2.Vpc(this, 'StreamlitVpc', {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
      ],
    });

    const albSecurityGroup = new ec2.SecurityGroup(this, 'StreamlitAlbSecurityGroup', {
      vpc,
      description: 'Internet facing ALB security group',
      allowAllOutbound: true,
    });
    albSecurityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'Allow HTTP');
    albSecurityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'Allow HTTPS');

    const securityGroup = new ec2.SecurityGroup(this, 'StreamlitSecurityGroup', {
      vpc,
      description: 'Allow inbound traffic to Streamlit',
      allowAllOutbound: true,
    });
    securityGroup.addIngressRule(
      ec2.Peer.securityGroupId(albSecurityGroup.securityGroupId),
      ec2.Port.tcp(port),
      'Allow traffic from ALB',
    );

    const role = new iam.Role(this, 'StreamlitInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
    });
    role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'));
    role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'));
    const logGroup = new logs.LogGroup(this, 'StreamlitLogGroup', {
      logGroupName: `/streamlit/${Stack.of(this).stackName}`,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:DescribeLogStreams',
          'logs:DescribeLogGroups',
        ],
        resources: [`${logGroup.logGroupArn}:*`, logGroup.logGroupArn],
      }),
    );

    const repo = ecr.Repository.fromRepositoryName(this, 'StreamlitRepo', props.repoName);
    const imageUriWithTag = `${repo.repositoryUri}:${imageTag}`;

    const region = Stack.of(this).region;
    const logGroupName = logGroup.logGroupName;
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'set -euxo pipefail',
      'dnf update -y',
      'dnf install -y docker',
      'systemctl enable --now docker',
      `aws logs create-log-group --log-group-name ${logGroupName} --region ${region} || true`,
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${repo.repositoryUri}`,
      `docker pull ${imageUriWithTag}`,
      'docker stop streamlit || true',
      'docker rm streamlit || true',
      'LOG_STREAM="instance-$(hostname)-$(date +%s)"',
      `docker run -d --restart unless-stopped -p ${port}:${port} -e LOCAL_JP_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc --log-driver awslogs --log-opt awslogs-group=${logGroupName} --log-opt awslogs-region=${region} --log-opt awslogs-create-group=false --log-opt awslogs-stream=$LOG_STREAM --name streamlit ${imageUriWithTag}`,
    );

    const instance = new ec2.Instance(this, 'StreamlitInstance', {
      vpc,
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup,
      role,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      associatePublicIpAddress: true,
      userData,
    });

    const alb = new elbv2.ApplicationLoadBalancer(this, 'StreamlitAlb', {
      vpc,
      internetFacing: true,
      securityGroup: albSecurityGroup,
    });

    const targetGroup = new elbv2.ApplicationTargetGroup(this, 'StreamlitTargetGroup', {
      vpc,
      port,
      protocol: elbv2.ApplicationProtocol.HTTP,
      protocolVersion: elbv2.ApplicationProtocolVersion.HTTP1,
      healthCheck: {
        path: '/',
        healthyHttpCodes: '200-399',
      },
    });
    targetGroup.addTarget(new elbv2Targets.InstanceIdTarget(instance.instanceId));

    let httpsCertificate: acm.ICertificate | undefined;
    let customDomain: string | undefined;

    if (props.hostedZoneDomain && props.subdomain) {
      const zone = route53.HostedZone.fromLookup(this, 'StreamlitHostedZone', {
        domainName: props.hostedZoneDomain,
      });
      customDomain = `${props.subdomain}.${props.hostedZoneDomain}`;
      httpsCertificate = new acm.DnsValidatedCertificate(this, 'StreamlitDnsValidatedCert', {
        domainName: customDomain,
        hostedZone: zone,
        region: Stack.of(this).region,
      });

      new route53.ARecord(this, 'StreamlitAliasRecord', {
        zone,
        recordName: customDomain,
        target: route53.RecordTarget.fromAlias(new route53Targets.LoadBalancerTarget(alb)),
      });
    } else if (props.certificateArn) {
      httpsCertificate = acm.Certificate.fromCertificateArn(
        this,
        'StreamlitCertificate',
        props.certificateArn,
      );
    }

    if (httpsCertificate) {
      alb.addListener('HttpsListener', {
        port: 443,
        certificates: [httpsCertificate],
        defaultAction: elbv2.ListenerAction.forward([targetGroup]),
      });
      alb.addListener('HttpRedirectListener', {
        port: 80,
        defaultAction: elbv2.ListenerAction.redirect({
          protocol: 'HTTPS',
          port: '443',
          permanent: true,
        }),
      });
    } else {
      alb.addListener('HttpListener', {
        port: 80,
        defaultAction: elbv2.ListenerAction.forward([targetGroup]),
      });
    }

    new CfnOutput(this, 'InstanceId', {
      value: instance.instanceId,
    });
    new CfnOutput(this, 'PublicDns', {
      value: instance.instancePublicDnsName,
    });
    new CfnOutput(this, 'PublicIp', {
      value: instance.instancePublicIp,
    });
    new CfnOutput(this, 'LoadBalancerDnsName', {
      value: alb.loadBalancerDnsName,
    });
    if (customDomain) {
      new CfnOutput(this, 'CustomDomainUrl', {
        value: `https://${customDomain}`,
      });
    }
  }
}
